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

/*
 * UDP receiver task.
 *
 * The AP controller sends UDP packets at a fixed rate. The receiver does not
 * use the packet content for classification yet. The main purpose of these
 * packets is to create controlled Wi-Fi traffic so the CSI callback is called
 * regularly.
 */
static void udp_receiver_task(void *pvParameters)
{
    char rx_buffer[128];

    /*
     * Listen on all local interfaces using the configured UDP port.
     */
    struct sockaddr_in listen_addr = {
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_family = AF_INET,
        .sin_port = htons(UDP_LISTEN_PORT),
    };

    /*
     * Creates a UDP socket.
     */
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create UDP socket");
        udp_receiver_started = false;
        vTaskDelete(NULL);
        return;
    }

    /*
     * Binds the socket to the selected port.
     * After this, incoming UDP packets sent to this port can be received.
     */
    int err = bind(sock, (struct sockaddr *)&listen_addr, sizeof(listen_addr));

    if (err < 0) {
        ESP_LOGE(TAG, "Failed to bind UDP socket");
        close(sock);
        udp_receiver_started = false;
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "UDP receiver started on port %d", UDP_LISTEN_PORT);

    /*
     * Main receive loop.
     *
     * recv() blocks until a UDP packet arrives. This is acceptable here because
     * this task has only one job: receive packets from the AP controller.
     */
    while (1) {
        int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);

        if (len > 0) {
            rx_buffer[len] = '\0';

            /*
             * UDP logs are intentionally disabled during CSI acquisition.
             *
             * Printing every UDP packet would flood the serial monitor and
             * disturb timing. The CSI callback is the relevant output.
             */
            // ESP_LOGI(TAG, "UDP packet received: %s", rx_buffer);
        }
    }
}

/*
 * Starts the UDP receiver task once.
 */
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