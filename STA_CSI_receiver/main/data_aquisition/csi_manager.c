#include "csi_manager.h"

#include <stdio.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_wifi_types.h"

static const char *TAG = "csi_manager";

static bool csi_enabled = false;

/*
 * CSI receive callback.
 *
 * This function is called by the Wi-Fi driver whenever a packet with CSI
 * information is received.
 *
 * Important:
 * This callback runs inside the Wi-Fi task context. For this reason, it must
 * remain lightweight. Heavy processing, filtering, feature extraction, or
 * classification should be moved to another task using a queue or buffer.
 */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    /*
     * Safety check.
     * info contains metadata and info->buf contains the raw CSI samples.
     */
    if (!info || !info->buf) {
        return;
    }

    /*
     * Temporary output format.
     *
     * At this stage, only basic metadata is printed. Later, this will be
     * expanded to include the raw CSI buffer so the data can be saved and
     * analyzed.
     */
    printf(
        "CSI,len=%d,rssi=%d,rate=%d,channel=%d\n",
        info->len,
        info->rx_ctrl.rssi,
        info->rx_ctrl.rate,
        info->rx_ctrl.channel
    );
}

/*
 * Enables CSI acquisition.
 *
 * CSI must be configured after Wi-Fi is initialized and after the station
 * connects to the AP. Calling this too early may cause esp_wifi_set_csi_config()
 * to fail.
 */
void csi_manager_start(void)
{
    /*
     * Avoid enabling CSI more than once after reconnection or repeated events.
     */
    if (csi_enabled) {
        return;
    }

    /*
     * Registers the callback that receives CSI samples.
     */
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));

    /*
     * CSI configuration.
     *
     * lltf_en:
     * Enables CSI extraction from Legacy Long Training Field.
     *
     * htltf_en:
     * Enables CSI extraction from High Throughput Long Training Field.
     *
     * stbc_htltf2_en:
     * Disabled for now to keep the acquisition simpler and avoid unsupported
     * configurations depending on packet format.
     *
     * ltf_merge_en:
     * Allows the driver to merge LTF information when available.
     *
     * channel_filter_en:
     * Disabled to preserve rawer CSI data for later preprocessing.
     *
     * manu_scale:
     * Disabled so the driver uses automatic scaling.
     */
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };

    /*
     * Applies CSI configuration and enables CSI collection.
     */
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    csi_enabled = true;

    ESP_LOGI(TAG, "CSI enabled");
}

/*
 * Resets CSI state after Wi-Fi disconnection.
 *
 * This does not fully deinitialize the Wi-Fi driver. It only allows the CSI
 * manager to run its setup again after reconnection.
 */
void csi_manager_reset(void)
{
    csi_enabled = false;
}