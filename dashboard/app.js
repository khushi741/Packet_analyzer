document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("inference-form");
    const resultBox = document.getElementById("prediction-result");
    const predictedApp = document.getElementById("predicted-app");
    const predictedConfBar = document.getElementById("predicted-conf-bar");
    const predictedConfText = document.getElementById("predicted-conf-text");
    const historyBody = document.getElementById("history-table-body");
    const statusDot = document.querySelector(".status-dot");
    const statusText = document.querySelector(".api-status span:last-child");

    // Dynamic API endpoints discovery
    const candidateEndpoints = [
        "/predict-flow",
        "http://127.0.0.1:8000/predict-flow",
        "http://127.0.0.1:8080/predict-flow",
        "http://localhost:8000/predict-flow"
    ];

    // Health check on startup to update status badge
    async function checkBackendHealth() {
        const healthEndpoints = [
            "/health",
            "http://127.0.0.1:8000/health",
            "http://127.0.0.1:8080/health",
            "http://localhost:8000/health"
        ];
        for (const ep of healthEndpoints) {
            try {
                const res = await fetch(ep, { method: "GET" });
                if (res.ok) {
                    const data = await res.json();
                    if (data.model_loaded) {
                        if (statusDot) statusDot.className = "status-dot green";
                        if (statusText) statusText.innerHTML = "ML Service: <strong>Online</strong>";
                        return;
                    }
                }
            } catch (e) {
                // Try next
            }
        }
        if (statusDot) statusDot.className = "status-dot yellow";
        if (statusText) statusText.innerHTML = "ML Service: <strong>Simulation Mode</strong>";
    }

    checkBackendHealth();

    // Initial Chart Setup
    const ctx = document.getElementById("trafficChart").getContext("2d");
    const trafficChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["YouTube", "Netflix", "Zoom", "Web Browsing", "File Download", "Social Media", "Unknown"],
            datasets: [{
                data: [312, 187, 96, 278, 140, 104, 45],
                backgroundColor: [
                    "rgba(0, 242, 254, 0.7)",  // youtube
                    "rgba(79, 172, 254, 0.7)",  // netflix
                    "rgba(127, 0, 255, 0.7)",   // zoom
                    "rgba(225, 0, 255, 0.7)",   // web browsing
                    "rgba(0, 230, 118, 0.7)",   // file download
                    "rgba(255, 23, 68, 0.7)",   // social media
                    "rgba(143, 160, 192, 0.7)"  // unknown
                ],
                borderColor: "rgba(255, 255, 255, 0.1)",
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "right",
                    labels: {
                        color: "#8fa0c0",
                        font: {
                            family: "Outfit"
                        }
                    }
                }
            }
        }
    });

    // Form Submit Handler
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const duration = parseFloat(document.getElementById("duration").value) || 1200;
        const packets = parseInt(document.getElementById("packets").value) || 42;
        const bytes = parseInt(document.getElementById("bytes").value) || 87000;
        const meanSize = parseFloat(document.getElementById("mean-size").value) || (bytes / packets);
        const iat = parseFloat(document.getElementById("iat").value) || 15;

        const durationSec = duration > 0 ? duration / 1000 : 1.0;
        const pps = packets / durationSec;
        const bps = bytes / durationSec;

        const payload = {
            flow_duration_ms: duration,
            total_packets: packets,
            total_bytes: bytes,
            forward_packets: Math.floor(packets * 0.4),
            backward_packets: packets - Math.floor(packets * 0.4),
            forward_bytes: Math.floor(bytes * 0.4),
            backward_bytes: bytes - Math.floor(bytes * 0.4),
            mean_packet_size: meanSize,
            std_packet_size: Math.round(meanSize * 0.25),
            min_packet_size: Math.max(40, Math.round(meanSize * 0.4)),
            max_packet_size: Math.min(1500, Math.round(meanSize * 1.5)),
            mean_inter_arrival_time_ms: iat,
            std_inter_arrival_time_ms: Math.round(iat * 0.3),
            packets_per_second: pps,
            bytes_per_second: bps,
            tcp_udp_flag: 6,
            detected_sni: "",
            dpi_app_type: "UNKNOWN"
        };

        let prediction = "unknown";
        let confidence = 0.5;
        let isAnomaly = false;
        let method = "SIM";

        // Try candidate endpoints
        let success = false;
        for (const ep of candidateEndpoints) {
            try {
                const response = await fetch(ep, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                if (response.ok) {
                    const data = await response.json();
                    prediction = data.prediction;
                    confidence = data.confidence;
                    isAnomaly = !!data.is_anomaly;
                    method = "ML";
                    success = true;
                    if (statusDot) statusDot.className = "status-dot green";
                    if (statusText) statusText.innerHTML = "ML Service: <strong>Online</strong>";
                    break;
                }
            } catch (err) {
                // Try next candidate
            }
        }

        // If backend offline, use realistic behavioral fallback
        if (!success) {
            if (meanSize > 900 && bytes > 50000) {
                prediction = Math.random() > 0.5 ? "youtube" : "netflix";
                confidence = 0.88 + Math.random() * 0.1;
            } else if (packets > 100 && iat < 40) {
                prediction = "zoom";
                confidence = 0.91 + Math.random() * 0.08;
            } else if (duration < 5000 && bytes < 30000) {
                prediction = "web_browsing";
                confidence = 0.75 + Math.random() * 0.15;
            } else {
                prediction = "social_media";
                confidence = 0.68 + Math.random() * 0.2;
            }
        }

        // Format prediction display name
        const appName = prediction.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');

        // Update UI result box
        predictedApp.textContent = isAnomaly ? `${appName} (Anomaly Detected!)` : appName;
        const pctConf = Math.round(confidence * 100);
        predictedConfBar.style.width = `${pctConf}%`;
        predictedConfText.textContent = `${pctConf}% Confidence`;
        resultBox.classList.remove("hidden");

        // Add to history table
        const newRow = document.createElement("tr");
        const badgeClass = method === "ML" ? "badge-ml" : "badge-dpi";
        newRow.innerHTML = `
            <td><span class="badge ${badgeClass}">${method}</span></td>
            <td><strong>${appName}</strong></td>
            <td>${(duration / 1000).toFixed(1)}s</td>
            <td>${(bytes / 1024).toFixed(1)} KB</td>
            <td>${pctConf}%</td>
        `;
        historyBody.insertBefore(newRow, historyBody.firstChild);

        // Update Chart statistics (increment count)
        const labelIndex = trafficChart.data.labels.indexOf(appName);
        if (labelIndex !== -1) {
            trafficChart.data.datasets[0].data[labelIndex]++;
            trafficChart.update();
        }
    });
});
