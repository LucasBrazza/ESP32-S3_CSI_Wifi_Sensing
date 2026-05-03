#include "udp_receiver.h"

#include <string.h>

#include "esp_log.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#define UDP_LISTEN_PORT 3333

static const char *TAG = "udp_receiver";

static bool udp_receiver_started = false;

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
        udp_receiver_started = false;
        vTaskDelete(NULL);
        return;
    }

    int err = bind(sock, (struct sockaddr *)&listen_addr, sizeof(listen_addr));

    if (err < 0) {
        ESP_LOGE(TAG, "Failed to bind UDP socket");
        close(sock);
        udp_receiver_started = false;
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "UDP receiver started on port %d", UDP_LISTEN_PORT);

    while (1) {
        int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);

        if (len > 0) {
            rx_buffer[len] = '\0';

            /*
             * UDP packets are intentionally not logged during CSI acquisition
             * to avoid serial flooding and timing interference.
             */
        }
    }
}

void udp_receiver_start(void)
{
    if (udp_receiver_started) {
        return;
    }

    udp_receiver_started = true;

    xTaskCreate(
        udp_receiver_task,
        "udp_receiver_task",
        4096,
        NULL,
        5,
        NULL
    );
}