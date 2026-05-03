#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi_types.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#define WIFI_SSID      CONFIG_WIFI_SSID
#define WIFI_PASSWORD  CONFIG_WIFI_PASSWORD

#define UDP_LISTEN_PORT 3333

static const char *TAG = "csi_receiver";

static bool csi_enabled = false;

/*
 * CSI callback.
 * Keep this function lightweight because it runs inside the Wi-Fi task context.
 */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        return;
    }

    printf(
        "CSI,len=%d,rssi=%d,rate=%d,channel=%d\n",
        info->len,
        info->rx_ctrl.rssi,
        info->rx_ctrl.rate,
        info->rx_ctrl.channel
    );
}

/*
 * Receives controlled UDP packets from the AP controller.
 * These packets generate repeatable Wi-Fi traffic for CSI acquisition.
 */
static void udp_receiver_task(void *pvParameters)
{
    char rx_buffer[128];

    struct sockaddr_in listen_addr = {
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_family = AF_INET,
        .sin_port = htons(UDP_LISTEN_PORT),
    };

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create UDP socket");
        vTaskDelete(NULL);
        return;
    }

    int err = bind(sock, (struct sockaddr *)&listen_addr, sizeof(listen_addr));

    if (err < 0) {
        ESP_LOGE(TAG, "Failed to bind UDP socket");
        close(sock);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "UDP receiver started on port %d", UDP_LISTEN_PORT);

    while (1) {
        int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);

        if (len > 0) {
            rx_buffer[len] = '\0';

            /*
             * Keep UDP logging disabled during CSI collection to avoid
             * flooding the serial output and disturbing timing.
             */
            // ESP_LOGI(TAG, "UDP packet received: %s", rx_buffer);
        }
    }
}

/*
 * Enables CSI acquisition on the station.
 */
static void csi_init(void)
{
    if (csi_enabled) {
        return;
    }

    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    csi_enabled = true;

    ESP_LOGI(TAG, "CSI enabled");
}

/*
 * Starts CSI and UDP reception after the station is fully connected.
 */
static void sensing_start_task(void *pvParameters)
{
    vTaskDelay(pdMS_TO_TICKS(1000));

    csi_init();

    xTaskCreate(
        udp_receiver_task,
        "udp_receiver_task",
        4096,
        NULL,
        5,
        NULL
    );

    vTaskDelete(NULL);
}

/*
 * Handles Wi-Fi and IP events.
 */
static void wifi_event_handler(
    void *arg,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data
)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Trying to connect to the access point...");
        esp_wifi_connect();
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Disconnected from AP. Reconnecting...");
        csi_enabled = false;
        esp_wifi_connect();
    }

    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *) event_data;

        ESP_LOGI(TAG, "Connected to AP");
        ESP_LOGI(TAG, "Assigned IP: " IPSTR, IP2STR(&event->ip_info.ip));

        xTaskCreate(
            sensing_start_task,
            "sensing_start_task",
            4096,
            NULL,
            5,
            NULL
        );
    }
}

/*
 * Initializes the ESP32-S3 as a Wi-Fi station.
 */
static void wifi_init_sta(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_sta();

    wifi_init_config_t wifi_init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init_config));

    ESP_ERROR_CHECK(esp_event_handler_register(
        WIFI_EVENT,
        ESP_EVENT_ANY_ID,
        &wifi_event_handler,
        NULL
    ));

    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT,
        IP_EVENT_STA_GOT_IP,
        &wifi_event_handler,
        NULL
    ));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi station started");
}

/*
 * Initializes the non-volatile storage used internally by the Wi-Fi stack.
 */
static void init_nvs(void)
{
    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
        return;
    }

    ESP_ERROR_CHECK(ret);
}

void app_main(void)
{
    init_nvs();
    wifi_init_sta();
}