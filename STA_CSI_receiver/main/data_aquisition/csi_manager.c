#include "csi_manager.h"

#include <stdio.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_wifi_types.h"

static const char *TAG = "csi_manager";

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

void csi_manager_start(void)
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

void csi_manager_reset(void)
{
    csi_enabled = false;
}