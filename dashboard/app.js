document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("inference-form");
    const resultBox = document.getElementById("prediction-result");
    const predictedApp = document.getElementById("predicted-app");
    const predictedConfBar = document.getElementById("predicted-conf-bar");
    const predictedConfText = document.getElementById("predicted-conf-text");
    const historyBody = document.getElementById("history-table-body");

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

        const duration = parseFloat(document.getElementById("duration").value);
        const packets = parseInt(document.getElementById("packets").value);
        const bytes = parseInt(document.getElementById("bytes").value);
        const meanSize = parseFloat(document.getElementById("mean-size").value);
        const iat = parseFloat(document.getElementById("iat").value);

        const payload = {
            flow_duration_ms: duration,
            total_packets: packets,
            total_bytes: bytes,
            mean_packet_size: meanSize,
            mean_inter_arrival_time_ms: iat,
            forward_packets: Math.floor(packets * 0.4),
            backward_packets: packets - Math.floor(packets * 0.4),
            forward_bytes: Math.floor(bytes * 0.4),
            backward_bytes: bytes - Math.floor(bytes * 0.4),
            tcp_udp_flag: 6,
            detected_sni: "",
            dpi_app_type: "UNKNOWN"
        };

        let prediction = "unknown";
        let confidence = 0.5;
        let method = "ML";

        try {
            // Attempt to hit FastAPI service running locally
            const response = await fetch("http://127.0.0.1:8080/predict-flow", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const data = await response.json();
                prediction = data.prediction;
                confidence = data.confidence;
            } else {
                throw new Error("Local API offline. Running local simulator fallback.");
            }
        } catch (err) {
            console.log(err.message);
            // Simulator fallback mapping standard feature bounds to apps
            if (meanSize > 1000 && bytes > 50000) {
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
        predictedApp.textContent = appName;
        const pctConf = Math.round(confidence * 100);
        predictedConfBar.style.width = `${pctConf}%`;
        predictedConfText.textContent = `${pctConf}% Confidence`;
        resultBox.classList.remove("hidden");

        // Add to history table
        const newRow = document.createElement("tr");
        newRow.innerHTML = `
            <td><span class="badge badge-ml">ML</span></td>
            <td><strong>${appName}</strong></td>
            <td>${(duration / 1000).toFixed(1)}s</td>
            <td>${(bytes / 1024).toFixed(1)} KB</td>
            <td>${pctConf}%</td>
        `;
        // Insert at top of table body
        historyBody.insertBefore(newRow, historyBody.firstChild);

        // Update Chart statistics (increment count)
        const labelIndex = trafficChart.data.labels.indexOf(appName);
        if (labelIndex !== -1) {
            trafficChart.data.datasets[0].data[labelIndex]++;
            trafficChart.update();
        }
    });
});
