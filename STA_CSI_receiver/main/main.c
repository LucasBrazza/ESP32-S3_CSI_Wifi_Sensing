#include "nvs_flash.h"
#include "esp_err.h"

#include "wifi_manager.h"

/*
 * Initializes NVS (Non-Volatile Storage).
 *
 * The ESP-IDF Wi-Fi stack uses NVS internally to store calibration data
 * and Wi-Fi-related information. Even if this application does not store
 * user data yet, NVS must be initialized before starting Wi-Fi.
 */
static void init_nvs(void)
{
    esp_err_t ret = nvs_flash_init();

    /*
     * If the NVS partition is full or was created with an incompatible
     * version, erase it and initialize again.
     */
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
        return;
    }

    ESP_ERROR_CHECK(ret);
}

/*
 * Application entry point.
 *
 * The main function is intentionally kept small. System logic is split into
 * dedicated modules to keep the project maintainable as CSI acquisition,
 * buffering, preprocessing, and embedded classification are added.
 */
void app_main(void)
{
    init_nvs();
    wifi_manager_start();
}