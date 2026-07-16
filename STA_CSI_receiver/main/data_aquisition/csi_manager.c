#include "csi_manager.h"
#include "esp_mac.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "driver/uart.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_idf_version.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_wifi_types.h"

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 3, 0)
#include "driver/uart_vfs.h"
#else
#include "esp_vfs_dev.h"
#endif

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"


static const char *TAG = "csi_manager";


/* ============================================================
 * UART TRANSPORT
 * ============================================================ */

/*
 * UART0 is the serial interface currently used by the project.
 *
 * At 921600 baud, the theoretical capacity is approximately
 * 92 kB/s using 8N1, which is sufficient for the current
 * target of approximately 50 CSI frames per second.
 */
#define CSI_UART_PORT UART_NUM_0
#define CSI_UART_BAUD_RATE 921600

/*
 * uart_driver_install() requires a valid RX ring buffer.
 *
 * Even though this application primarily transmits CSI data,
 * the RX buffer cannot be zero when installing the UART driver.
 */
#define CSI_UART_RX_BUFFER_SIZE 1024
#define CSI_UART_TX_BUFFER_SIZE 16384


/* ============================================================
 * CSI ACQUISITION BUFFERING
 * ============================================================ */

#define CSI_MAX_LEN 384
#define CSI_QUEUE_LENGTH 64

#define CSI_OUTPUT_TASK_STACK_SIZE 4096
#define CSI_OUTPUT_TASK_PRIORITY 6


/* ============================================================
 * BINARY PROTOCOL
 * ============================================================ */

#define CSI_PROTOCOL_VERSION 1

#define CSI_FRAME_TYPE_SAMPLE 1
#define CSI_FRAME_TYPE_STATS 2

#define CSI_MAGIC_SIZE 4
#define CSI_COMMON_HEADER_SIZE 8
#define CSI_SAMPLE_METADATA_SIZE 18

#define CSI_SAMPLE_FRAME_MAX_SIZE \
    (CSI_COMMON_HEADER_SIZE \
     + CSI_SAMPLE_METADATA_SIZE \
     + CSI_MAX_LEN \
     + 2)

#define CSI_STATS_FRAME_SIZE 46
#define CSI_STATS_INTERVAL_US 1000000LL


static const uint8_t CSI_MAGIC[CSI_MAGIC_SIZE] = {
    'C',
    'S',
    'I',
    '2'
};


/* ============================================================
 * DATA TYPES
 * ============================================================ */

typedef struct {
    uint32_t sequence;
    int64_t timestamp_us;

    int8_t rssi;
    uint8_t rate;
    uint8_t channel;
    uint8_t flags;

    uint16_t len;
    int8_t data[CSI_MAX_LEN];
} csi_sample_t;

/* ============================================================
 * INTERNAL STATE
 * ============================================================ */

static bool csi_enabled = false;

static QueueHandle_t csi_queue = NULL;

static TaskHandle_t csi_output_task_handle = NULL;


/*
 * MAC addresses used to accept only controlled unicast traffic
 * sent by the configured AP directly to this STA.
 */
static uint8_t expected_ap_bssid[6] = {0};

static uint8_t expected_sta_mac[6] = {0};

static bool csi_mac_filter_ready = false;



/*
 * Diagnostic counters.
 *
 * These counters are periodically sent to the computer through
 * CSI_FRAME_TYPE_STATS frames.
 */
static volatile uint32_t csi_received_count = 0;
static volatile uint32_t csi_queued_count = 0;
static volatile uint32_t csi_serialized_count = 0;
static volatile uint32_t csi_queue_drop_count = 0;
static volatile uint32_t csi_invalid_count = 0;
static volatile uint32_t csi_oversize_count = 0;


/* ============================================================
 * LITTLE-ENDIAN SERIALIZATION
 * ============================================================ */

static void write_u16_le(
    uint8_t *destination,
    uint16_t value
)
{
    destination[0] = (uint8_t)(value & 0xFFU);
    destination[1] = (uint8_t)((value >> 8) & 0xFFU);
}


static void write_u32_le(
    uint8_t *destination,
    uint32_t value
)
{
    destination[0] = (uint8_t)(value & 0xFFU);
    destination[1] = (uint8_t)((value >> 8) & 0xFFU);
    destination[2] = (uint8_t)((value >> 16) & 0xFFU);
    destination[3] = (uint8_t)((value >> 24) & 0xFFU);
}


static void write_u64_le(
    uint8_t *destination,
    uint64_t value
)
{
    for (uint8_t index = 0; index < 8; index++) {
        destination[index] = (uint8_t)(
            value >> (8U * index)
        );
    }
}


/* ============================================================
 * CRC
 * ============================================================ */

/*
 * CRC-16/CCITT-FALSE
 *
 * Polynomial:    0x1021
 * Initial value: 0xFFFF
 */
static uint16_t calculate_crc16_ccitt(
    const uint8_t *data,
    size_t length
)
{
    uint16_t crc = 0xFFFFU;

    for (
        size_t byte_index = 0;
        byte_index < length;
        byte_index++
    ) {
        crc ^= (uint16_t)data[byte_index] << 8;

        for (
            uint8_t bit_index = 0;
            bit_index < 8;
            bit_index++
        ) {
            if ((crc & 0x8000U) != 0U) {
                crc = (uint16_t)(
                    (crc << 1) ^ 0x1021U
                );
            } else {
                crc <<= 1;
            }
        }
    }

    return crc;
}


/* ============================================================
 * FRAME CONSTRUCTION
 * ============================================================ */

static void write_common_header(
    uint8_t *frame,
    uint8_t frame_type,
    uint16_t frame_size
)
{
    memcpy(
        frame,
        CSI_MAGIC,
        CSI_MAGIC_SIZE
    );

    frame[4] = CSI_PROTOCOL_VERSION;
    frame[5] = frame_type;

    write_u16_le(
        frame + 6,
        frame_size
    );
}


static size_t build_sample_frame(
    const csi_sample_t *sample,
    uint8_t *frame,
    size_t frame_capacity
)
{
    if (sample == NULL || frame == NULL) {
        return 0;
    }

    const size_t frame_size =
        CSI_COMMON_HEADER_SIZE
        + CSI_SAMPLE_METADATA_SIZE
        + sample->len
        + 2;

    if (
        frame_size > frame_capacity
        || frame_size > UINT16_MAX
    ) {
        return 0;
    }

    write_common_header(
        frame,
        CSI_FRAME_TYPE_SAMPLE,
        (uint16_t)frame_size
    );

    size_t offset = CSI_COMMON_HEADER_SIZE;

    write_u32_le(
        frame + offset,
        sample->sequence
    );
    offset += 4;

    write_u64_le(
        frame + offset,
        (uint64_t)sample->timestamp_us
    );
    offset += 8;

    frame[offset++] = (uint8_t)sample->rssi;
    frame[offset++] = sample->rate;
    frame[offset++] = sample->channel;
    frame[offset++] = sample->flags;

    write_u16_le(
        frame + offset,
        sample->len
    );
    offset += 2;

    memcpy(
        frame + offset,
        sample->data,
        sample->len
    );
    offset += sample->len;

    /*
     * The CRC covers the frame starting at the protocol version.
     * The CSI2 magic is excluded.
     */
    const uint16_t crc = calculate_crc16_ccitt(
        frame + CSI_MAGIC_SIZE,
        offset - CSI_MAGIC_SIZE
    );

    write_u16_le(
        frame + offset,
        crc
    );
    offset += 2;

    return offset;
}


static size_t build_stats_frame(
    uint8_t *frame,
    size_t frame_capacity
)
{
    if (
        frame == NULL
        || frame_capacity < CSI_STATS_FRAME_SIZE
    ) {
        return 0;
    }

    write_common_header(
        frame,
        CSI_FRAME_TYPE_STATS,
        CSI_STATS_FRAME_SIZE
    );

    size_t offset = CSI_COMMON_HEADER_SIZE;

    write_u64_le(
        frame + offset,
        (uint64_t)esp_timer_get_time()
    );
    offset += 8;

    write_u32_le(
        frame + offset,
        csi_received_count
    );
    offset += 4;

    write_u32_le(
        frame + offset,
        csi_queued_count
    );
    offset += 4;

    write_u32_le(
        frame + offset,
        csi_serialized_count
    );
    offset += 4;

    write_u32_le(
        frame + offset,
        csi_queue_drop_count
    );
    offset += 4;

    write_u32_le(
        frame + offset,
        csi_invalid_count
    );
    offset += 4;

    write_u32_le(
        frame + offset,
        csi_oversize_count
    );
    offset += 4;

    const uint16_t queue_pending =
        csi_queue == NULL
        ? 0
        : (uint16_t)uxQueueMessagesWaiting(
            csi_queue
        );

    write_u16_le(
        frame + offset,
        queue_pending
    );
    offset += 2;

    /*
     * Reserved for future statistics.
     */
    write_u16_le(
        frame + offset,
        0
    );
    offset += 2;

    const uint16_t crc = calculate_crc16_ccitt(
        frame + CSI_MAGIC_SIZE,
        offset - CSI_MAGIC_SIZE
    );

    write_u16_le(
        frame + offset,
        crc
    );
    offset += 2;

    return offset;
}


/* ============================================================
 * UART TRANSMISSION
 * ============================================================ */

static bool send_binary_frame(
    const uint8_t *frame,
    size_t frame_size
)
{
    if (
        frame == NULL
        || frame_size == 0
    ) {
        return false;
    }

    size_t total_written = 0;

    while (total_written < frame_size) {
        const int written = uart_write_bytes(
            CSI_UART_PORT,
            frame + total_written,
            frame_size - total_written
        );

        if (written <= 0) {
            return false;
        }

        total_written += (size_t)written;
    }

    return true;
}


/*
 * Initializes UART0 for binary CSI transmission.
 *
 * The UART driver is installed before changing the baud rate.
 * This avoids leaving the console at 921600 baud if driver
 * installation fails.
 */
static esp_err_t initialize_serial_transport(void)
{
    /*
     * Install the UART driver only if it is not already installed.
     */
    if (!uart_is_driver_installed(CSI_UART_PORT)) {
        ESP_RETURN_ON_ERROR(
            uart_driver_install(
                CSI_UART_PORT,
                CSI_UART_RX_BUFFER_SIZE,
                CSI_UART_TX_BUFFER_SIZE,
                0,
                NULL,
                0
            ),
            TAG,
            "Failed to install CSI UART driver"
        );
    }

    const uart_config_t uart_config = {
        .baud_rate = CSI_UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_RETURN_ON_ERROR(
        uart_param_config(
            CSI_UART_PORT,
            &uart_config
        ),
        TAG,
        "Failed to configure CSI UART"
    );

    /*
     * Route stdout and ESP-IDF logs through the buffered UART driver.
     *
     * The Python parser searches for the CSI2 magic and can ignore
     * textual startup messages that precede the binary frames.
     */
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 3, 0)
    uart_vfs_dev_use_driver(CSI_UART_PORT);
#else
    esp_vfs_dev_uart_use_driver(CSI_UART_PORT);
#endif

    return ESP_OK;
}


/* ============================================================
 * OUTPUT TASK
 * ============================================================ */

static void csi_output_task(void *parameter)
{
    (void)parameter;

    csi_sample_t sample;

    uint8_t frame[CSI_SAMPLE_FRAME_MAX_SIZE];

    int64_t last_stats_time_us =
        esp_timer_get_time();

    while (true) {
        const BaseType_t received = xQueueReceive(
            csi_queue,
            &sample,
            pdMS_TO_TICKS(20)
        );

        if (received == pdTRUE) {
            const size_t frame_size =
                build_sample_frame(
                    &sample,
                    frame,
                    sizeof(frame)
                );

            if (
                frame_size == 0
                || !send_binary_frame(
                    frame,
                    frame_size
                )
            ) {
                csi_invalid_count++;
            } else {
                csi_serialized_count++;
            }
        }

        const int64_t current_time_us =
            esp_timer_get_time();

        if (
            current_time_us - last_stats_time_us
            >= CSI_STATS_INTERVAL_US
        ) {
            const size_t stats_size =
                build_stats_frame(
                    frame,
                    sizeof(frame)
                );

            if (stats_size > 0) {
                if (!send_binary_frame(
                        frame,
                        stats_size
                    )) {
                    csi_invalid_count++;
                }
            }

            last_stats_time_us =
                current_time_us;
        }
    }
}


/* ============================================================
 * PIPELINE INITIALIZATION
 * ============================================================ */

static esp_err_t initialize_csi_pipeline(void)
{
    if (csi_queue == NULL) {
        csi_queue = xQueueCreate(
            CSI_QUEUE_LENGTH,
            sizeof(csi_sample_t)
        );

        if (csi_queue == NULL) {
            ESP_LOGE(
                TAG,
                "Failed to create CSI queue"
            );

            return ESP_ERR_NO_MEM;
        }
    }

    ESP_RETURN_ON_ERROR(
        initialize_serial_transport(),
        TAG,
        "Failed to initialize serial transport"
    );

    if (csi_output_task_handle == NULL) {
        const BaseType_t task_created =
            xTaskCreate(
                csi_output_task,
                "csi_output_task",
                CSI_OUTPUT_TASK_STACK_SIZE,
                NULL,
                CSI_OUTPUT_TASK_PRIORITY,
                &csi_output_task_handle
            );

        if (task_created != pdPASS) {
            ESP_LOGE(
                TAG,
                "Failed to create CSI output task"
            );

            return ESP_ERR_NO_MEM;
        }
    }

    return ESP_OK;
}


/* ============================================================
 * WIFI CSI CALLBACK
 * ============================================================ */

/*
 * This callback runs in the Wi-Fi task context.
 *
 * The CSI buffer belongs to the Wi-Fi driver and becomes invalid
 * after this callback returns. Therefore, it must be copied
 * immediately.
 */


static esp_err_t initialize_csi_mac_filter(void)
{
    wifi_ap_record_t ap_info = {0};

    ESP_RETURN_ON_ERROR(
        esp_wifi_sta_get_ap_info(&ap_info),
        TAG,
        "Failed to obtain connected AP information"
    );

    ESP_RETURN_ON_ERROR(
        esp_wifi_get_mac(
            WIFI_IF_STA,
            expected_sta_mac
        ),
        TAG,
        "Failed to obtain STA MAC address"
    );

    memcpy(
        expected_ap_bssid,
        ap_info.bssid,
        sizeof(expected_ap_bssid)
    );

    csi_mac_filter_ready = true;

    ESP_LOGI(
        TAG,
        "CSI MAC filter: AP=" MACSTR ", STA=" MACSTR,
        MAC2STR(expected_ap_bssid),
        MAC2STR(expected_sta_mac)
    );

    return ESP_OK;
}


static void wifi_csi_rx_cb(
    void *ctx,
    wifi_csi_info_t *info
)
{
    (void)ctx;

    if (
        info == NULL
        || info->buf == NULL
        || info->len <= 0
    ) {
        csi_invalid_count++;
        return;
    }

    /*
     * Ignore beacons, broadcasts and frames from unrelated devices.
     *
     * The controlled UDP packet must:
     * - originate from the AP BSSID;
     * - be addressed directly to this STA.
     */
    if (
        !csi_mac_filter_ready
        || memcmp(
            info->mac,
            expected_ap_bssid,
            sizeof(expected_ap_bssid)
        ) != 0
        || memcmp(
            info->dmac,
            expected_sta_mac,
            sizeof(expected_sta_mac)
        ) != 0
    ) {
        return;
    }

    if (info->len > CSI_MAX_LEN) {
        csi_oversize_count++;
        return;
    }

    /*
     * Count only frames that passed the controlled-traffic filter.
     */
    csi_received_count++;

    /*
     * Flags layout:
     *
     * bit 0     : channel width, 0=20 MHz, 1=40 MHz
     * bits 1-2  : signal mode
     * bit 3     : STBC enabled
     * bit 4     : first CSI word invalid
     * bits 5-7  : MCS, lower three bits
     */
    uint8_t flags = 0;

    flags |= (uint8_t)(
        info->rx_ctrl.cwb & 0x01U
    );

    flags |= (uint8_t)(
        (info->rx_ctrl.sig_mode & 0x03U) << 1
    );

    if (info->rx_ctrl.stbc != 0U) {
        flags |= (1U << 3);
    }

    if (info->first_word_invalid) {
        flags |= (1U << 4);
    }

    flags |= (uint8_t)(
        (info->rx_ctrl.mcs & 0x07U) << 5
    );

    csi_sample_t sample = {
        .sequence = csi_received_count,
        .timestamp_us = esp_timer_get_time(),
        .rssi = info->rx_ctrl.rssi,
        .rate = info->rx_ctrl.rate,
        .channel = info->rx_ctrl.channel,
        .flags = flags,
        .len = (uint16_t)info->len,
    };

    memcpy(
        sample.data,
        info->buf,
        sample.len
    );

    if (
        xQueueSend(
            csi_queue,
            &sample,
            0
        ) == pdTRUE
    ) {
        csi_queued_count++;
    } else {
        csi_queue_drop_count++;
    }
}

/* ============================================================
 * PUBLIC API
 * ============================================================ */

void csi_manager_start(void)
{
    if (csi_enabled) {
        return;
    }

    /*
     * Obtain the connected AP BSSID and this STA MAC address.
     *
     * These addresses are used by the CSI callback to accept only
     * controlled unicast traffic sent by the AP to this STA.
     */
    esp_err_t result =
        initialize_csi_mac_filter();

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "Failed to initialize CSI MAC filter: %s",
            esp_err_to_name(result)
        );

        return;
    }

    /*
     * Initialize the CSI queue, UART binary transport and output task.
     */
    result = initialize_csi_pipeline();

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "CSI pipeline initialization failed: %s",
            esp_err_to_name(result)
        );

        return;
    }

    result = esp_wifi_set_csi_rx_cb(
        wifi_csi_rx_cb,
        NULL
    );

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "Failed to register CSI callback: %s",
            esp_err_to_name(result)
        );

        return;
    }

    const wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };

    result = esp_wifi_set_csi_config(
        &csi_config
    );

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "Failed to configure CSI: %s",
            esp_err_to_name(result)
        );

        return;
    }

    ESP_LOGI(
        TAG,
        "Starting binary CSI stream at %d baud",
        CSI_UART_BAUD_RATE
    );

    /*
     * Wait for the final textual log before sending binary frames.
     */
    (void)uart_wait_tx_done(
        CSI_UART_PORT,
        pdMS_TO_TICKS(200)
    );

    result = esp_wifi_set_csi(true);

    if (result != ESP_OK) {
        ESP_LOGE(
            TAG,
            "Failed to enable CSI: %s",
            esp_err_to_name(result)
        );

        return;
    }

    csi_enabled = true;
}


void csi_manager_reset(void)
{
    if (csi_enabled) {
        const esp_err_t result =
            esp_wifi_set_csi(false);

        if (result != ESP_OK) {
            ESP_LOGW(
                TAG,
                "Failed to disable CSI: %s",
                esp_err_to_name(result)
            );
        }
    }

    csi_enabled = false;
    csi_mac_filter_ready = false;

    if (csi_queue != NULL) {
        xQueueReset(csi_queue);
    }
}