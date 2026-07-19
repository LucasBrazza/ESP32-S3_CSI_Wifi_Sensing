#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"


#define WIFI_SSID      CONFIG_WIFI_SSID
#define WIFI_PASSWORD  CONFIG_WIFI_PASSWORD
#define WIFI_CHANNEL   CONFIG_WIFI_CHANNEL
#define MAX_CLIENTS    CONFIG_MAX_CLIENTS

#define UDP_TARGET_IP       "192.168.4.2"
#define UDP_TARGET_PORT     3333
#define UDP_INTERVAL_MS     20
#define UDP_STATS_PERIOD_US 1000000LL


static const char *TAG = "ap_controller";


/*
 * Sends controlled UDP packets to the CSI receiver.
 *
 * xTaskDelayUntil() keeps the transmission period referenced to
 * an absolute wake-up time. Unlike a simple delay after sendto(),
 * the execution time of snprintf() and sendto() is not added to
 * every 20 ms interval.
 */
static void udp_traffic_task(void *pv_parameters)
{
    (void)pv_parameters;

    const int sock = socket(
        AF_INET,
        SOCK_DGRAM,
        IPPROTO_IP
    );

    if (sock < 0) {
        ESP_LOGE(
            TAG,
            "Failed to create UDP socket"
        );

        vTaskDelete(NULL);
        return;
    }

    const struct sockaddr_in destination = {
        .sin_addr.s_addr = inet_addr(UDP_TARGET_IP),
        .sin_family = AF_INET,
        .sin_port = htons(UDP_TARGET_PORT),
    };

    char payload[64];

    uint32_t packet_sequence = 0;
    uint32_t packets_sent = 0;
    uint32_t send_errors = 0;

    uint32_t previous_sent_count = 0;
    uint32_t previous_error_count = 0;

    int64_t previous_stats_time_us =
        esp_timer_get_time();

    TickType_t previous_wake_time =
        xTaskGetTickCount();

    const TickType_t period_ticks =
        pdMS_TO_TICKS(UDP_INTERVAL_MS);

    ESP_LOGI(
        TAG,
        "UDP traffic task started"
    );

    ESP_LOGI(
        TAG,
        "Target: %s:%d",
        UDP_TARGET_IP,
        UDP_TARGET_PORT
    );

    ESP_LOGI(
        TAG,
        "Requested packet interval: %d ms",
        UDP_INTERVAL_MS
    );

    while (true) {
        const int64_t timestamp_us =
            esp_timer_get_time();

        const int payload_length = snprintf(
            payload,
            sizeof(payload),
            "CSI_PKT,%" PRIu32 ",%" PRId64,
            packet_sequence,
            timestamp_us
        );

        if (
            payload_length <= 0
            || payload_length >= (int)sizeof(payload)
        ) {
            ESP_LOGE(
                TAG,
                "Failed to build UDP payload"
            );

            send_errors++;
        } else {
            const int sent = sendto(
                sock,
                payload,
                payload_length,
                0,
                (const struct sockaddr *)&destination,
                sizeof(destination)
            );

            if (sent == payload_length) {
                packets_sent++;
                packet_sequence++;
            } else {
                send_errors++;
            }
        }

        const int64_t current_time_us =
            esp_timer_get_time();

        const int64_t stats_elapsed_us =
            current_time_us - previous_stats_time_us;

        if (stats_elapsed_us >= UDP_STATS_PERIOD_US) {
            const uint32_t sent_during_period =
                packets_sent - previous_sent_count;

            const uint32_t errors_during_period =
                send_errors - previous_error_count;

            const float elapsed_seconds =
                (float)stats_elapsed_us / 1000000.0F;

            const float send_rate =
                elapsed_seconds > 0.0F
                ? (float)sent_during_period / elapsed_seconds
                : 0.0F;

            ESP_LOGI(
                TAG,
                "UDP stats: rate=%.2f pkt/s, sent=%" PRIu32
                ", errors=%" PRIu32,
                send_rate,
                sent_during_period,
                errors_during_period
            );

            previous_sent_count = packets_sent;
            previous_error_count = send_errors;
            previous_stats_time_us = current_time_us;
        }

        /*
         * ESP-IDF 6.0 uses xTaskDelayUntil().
         */
        (void)xTaskDelayUntil(
            &previous_wake_time,
            period_ticks
        );
    }
}


/*
 * Initializes the ESP32-S3 as a controlled Wi-Fi access point.
 */
static void wifi_init_ap(void)
{
    ESP_ERROR_CHECK(
        esp_netif_init()
    );

    ESP_ERROR_CHECK(
        esp_event_loop_create_default()
    );

    esp_netif_create_default_wifi_ap();

    const wifi_init_config_t wifi_init_config =
        WIFI_INIT_CONFIG_DEFAULT();

    ESP_ERROR_CHECK(
        esp_wifi_init(&wifi_init_config)
    );

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
        wifi_config.ap.authmode =
            WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(
        esp_wifi_set_mode(WIFI_MODE_AP)
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_config(
            WIFI_IF_AP,
            &wifi_config
        )
    );

    ESP_ERROR_CHECK(
        esp_wifi_start()
    );

    /*
     * Force HT20 operation.
     *
     * HT40 may fall back dynamically to 20 MHz depending on
     * coexistence and radio conditions, producing CSI vectors
     * with different lengths. HT20 provides a fixed and more
     * reproducible CSI shape for the dataset.
     *
     * The interface must already be enabled before calling
     * esp_wifi_set_bandwidth().
     */
    ESP_ERROR_CHECK(
        esp_wifi_set_bandwidth(
            WIFI_IF_AP,
            WIFI_BW20
        )
    );

    ESP_LOGI(
        TAG,
        "Wi-Fi access point started"
    );

    ESP_LOGI(
        TAG,
        "SSID: %s",
        WIFI_SSID
    );

    ESP_LOGI(
        TAG,
        "Channel: %d",
        WIFI_CHANNEL
    );

    ESP_LOGI(
        TAG,
        "Bandwidth: HT20"
    );

    ESP_LOGI(
        TAG,
        "Maximum clients: %d",
        MAX_CLIENTS
    );

    const BaseType_t task_created =
        xTaskCreate(
            udp_traffic_task,
            "udp_traffic_task",
            4096,
            NULL,
            5,
            NULL
        );

    if (task_created != pdPASS) {
        ESP_LOGE(
            TAG,
            "Failed to create UDP traffic task"
        );
    }
}


/*
 * Initializes NVS used internally by the Wi-Fi stack.
 */
static void init_nvs(void)
{
    esp_err_t result =
        nvs_flash_init();

    if (
        result == ESP_ERR_NVS_NO_FREE_PAGES
        || result == ESP_ERR_NVS_NEW_VERSION_FOUND
    ) {
        ESP_ERROR_CHECK(
            nvs_flash_erase()
        );

        ESP_ERROR_CHECK(
            nvs_flash_init()
        );

        return;
    }

    ESP_ERROR_CHECK(result);
}


void app_main(void)
{
    init_nvs();
    wifi_init_ap();
}