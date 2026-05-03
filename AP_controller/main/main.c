#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#define WIFI_SSID      CONFIG_WIFI_SSID
#define WIFI_PASSWORD  CONFIG_WIFI_PASSWORD
#define WIFI_CHANNEL   CONFIG_WIFI_CHANNEL
#define MAX_CLIENTS    CONFIG_MAX_CLIENTS

#define UDP_TARGET_IP      "192.168.4.2"
#define UDP_TARGET_PORT    3333
#define UDP_INTERVAL_MS    20

static const char *TAG = "ap_controller";

/*
 * Sends controlled UDP packets to the CSI receiver.
 * A fixed packet rate is important for repeatable CSI sensing experiments.
 */
static void udp_traffic_task(void *pvParameters)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create UDP socket");
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in dest_addr = {
        .sin_addr.s_addr = inet_addr(UDP_TARGET_IP),
        .sin_family = AF_INET,
        .sin_port = htons(UDP_TARGET_PORT),
    };

    uint32_t packet_counter = 0;
    char payload[64];

    ESP_LOGI(TAG, "UDP traffic task started");
    ESP_LOGI(TAG, "Target: %s:%d", UDP_TARGET_IP, UDP_TARGET_PORT);
    ESP_LOGI(TAG, "Packet interval: %d ms", UDP_INTERVAL_MS);

    while (1) {
        int64_t timestamp_us = esp_timer_get_time();

        int payload_len = snprintf(
            payload,
            sizeof(payload),
            "CSI_PKT,%" PRIu32 ",%" PRId64,
            packet_counter++,
            timestamp_us
        );

        int sent = sendto(
            sock,
            payload,
            payload_len,
            0,
            (struct sockaddr *)&dest_addr,
            sizeof(dest_addr)
        );

        if (sent < 0) {
            ESP_LOGW(TAG, "Failed to send UDP packet");
        }

        vTaskDelay(pdMS_TO_TICKS(UDP_INTERVAL_MS));
    }
}

/*
 * Initializes the ESP32-S3 as a Wi-Fi access point.
 * This node acts as the controlled transmitter for CSI sensing.
 */
static void wifi_init_ap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_create_default_wifi_ap();

    wifi_init_config_t wifi_init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init_config));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = WIFI_SSID,
            .ssid_len = 0,
            .channel = WIFI_CHANNEL,
            .password = WIFI_PASSWORD,
            .max_connection = MAX_CLIENTS,
            .authmode = WIFI_AUTH_WPA_WPA2_PSK,
        },
    };

    if (strlen(WIFI_PASSWORD) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    /*
     * Fixed channel operation is required for consistent CSI measurements.
     */
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi access point started");
    ESP_LOGI(TAG, "SSID: %s", WIFI_SSID);
    ESP_LOGI(TAG, "Channel: %d", WIFI_CHANNEL);
    ESP_LOGI(TAG, "Maximum clients: %d", MAX_CLIENTS);

    xTaskCreate(
        udp_traffic_task,
        "udp_traffic_task",
        4096,
        NULL,
        5,
        NULL
    );
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
    wifi_init_ap();
}