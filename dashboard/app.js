// ==============================================================================
// Unified Traffic Analyzer & ML Inference Dashboard Logic
// ==============================================================================

// Global Chart Instances & Data Cache
let classChartInstance = null;
let dpiMlChartInstance = null;
let throughputChartInstance = null;
let featureChartInstance = null;
let fullDatasetCache = [];

// Flow Scenario Presets for Calculator
const presets = {
    youtube: { duration: 35000, packets: 1450, bytes: 1850000, mean: 1275, iat: 24 },
    zoom: { duration: 60000, packets: 2800, bytes: 420000, mean: 150, iat: 20 },
    download: { duration: 15000, packets: 8500, bytes: 12500000, mean: 1470, iat: 2 },
    web: { duration: 3200, packets: 38, bytes: 28000, mean: 736, iat: 84 }
};

// ==============================================================================
// 1. VIEW SWITCHING (Data Showcase vs Flow Calculator)
// ==============================================================================
function switchView(view) {
    const btnTabData = document.getElementById("btn-tab-data");
    const btnTabCalc = document.getElementById("btn-tab-calc");
    const viewData = document.getElementById("view-data");
    const viewCalc = document.getElementById("view-calc");

    if (!btnTabData || !btnTabCalc || !viewData || !viewCalc) return;

    if (view === "calc") {
        btnTabCalc.classList.add("active");
        btnTabData.classList.remove("active");
        viewCalc.classList.remove("view-hidden");
        viewData.classList.add("view-hidden");
        if (window.location.hash !== "#calculate" && window.location.hash !== "#calc") {
            history.replaceState(null, null, "#calculate");
        }
    } else {
        btnTabData.classList.add("active");
        btnTabCalc.classList.remove("active");
        viewData.classList.remove("view-hidden");
        viewCalc.classList.add("view-hidden");
        if (window.location.hash !== "#data") {
            history.replaceState(null, null, "#data");
        }
        if (!fullDatasetCache || fullDatasetCache.length === 0) {
            fetchAnalyticsData();
        }
    }
}
window.switchView = switchView;

// ==============================================================================
// 2. DATA SHOWCASE: FETCH & RENDER ANALYTICS
// ==============================================================================
async function fetchAnalyticsData() {
    const refreshBtn = document.querySelector(".preset-btn[onclick*='fetchAnalyticsData']");
    if (refreshBtn) refreshBtn.textContent = "⏳ Refreshing...";

    try {
        const res = await fetch("/api/analytics");
        if (!res.ok) throw new Error("Analytics API returned HTTP " + res.status);
        const data = await res.json();

        if (!data || !data.available) {
            document.getElementById("stat-total-flows").textContent = "0";
            document.getElementById("stat-total-packets").textContent = "0";
            document.getElementById("stat-total-volume").textContent = "0 MB";
            document.getElementById("stat-avg-duration").textContent = "0s";
            return;
        }

        // 1. Update 4 Hero Metric Cards immediately
        const flowsEl = document.getElementById("stat-total-flows");
        const packetsEl = document.getElementById("stat-total-packets");
        const volumeEl = document.getElementById("stat-total-volume");
        const durationEl = document.getElementById("stat-avg-duration");

        if (flowsEl) flowsEl.textContent = Number(data.total_flows).toLocaleString();
        if (packetsEl) packetsEl.textContent = Number(data.total_packets).toLocaleString();
        if (volumeEl) volumeEl.textContent = `${data.total_mb} MB`;
        if (durationEl) durationEl.textContent = `${data.avg_duration_sec}s`;

        // 2. Populate Captured Flows Dataset Table
        fullDatasetCache = data.sample_flows || [];
        renderDatasetTable(fullDatasetCache);

        // 3. Render Visualizations Safely with Chart.js
        if (typeof Chart !== "undefined") {
            try { renderClassChart(data.class_distribution || {}); } catch (e) { console.error("Class chart error:", e); }
            try { renderDpiMlChart(data.dpi_count || 0, data.ml_count || 0); } catch (e) { console.error("DPI/ML chart error:", e); }
            try { renderThroughputChart(data.throughput_buckets || {}); } catch (e) { console.error("Throughput chart error:", e); }
            try { renderFeatureChart(data.feature_importances || []); } catch (e) { console.error("Feature chart error:", e); }
        } else {
            console.warn("Chart.js is not yet available; charts skipped.");
        }

    } catch (err) {
        console.error("Failed to load analytics data:", err);
        const flowsEl = document.getElementById("stat-total-flows");
        if (flowsEl && flowsEl.textContent === "...") {
            flowsEl.textContent = "N/A";
        }
    } finally {
        if (refreshBtn) refreshBtn.textContent = "🔄 Refresh Dataset";
    }
}
window.fetchAnalyticsData = fetchAnalyticsData;

// ==============================================================================
// 3. CHART.JS VISUALIZATION RENDERERS
// ==============================================================================
function renderClassChart(classDist) {
    const canvas = document.getElementById("classChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const labels = Object.keys(classDist);
    const counts = Object.values(classDist);

    if (classChartInstance) {
        classChartInstance.destroy();
    }

    classChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: [
                    "rgba(0, 242, 254, 0.75)",
                    "rgba(79, 172, 254, 0.75)",
                    "rgba(127, 0, 255, 0.75)",
                    "rgba(225, 0, 255, 0.75)",
                    "rgba(0, 230, 118, 0.75)",
                    "rgba(255, 23, 68, 0.75)",
                ],
                borderColor: "rgba(255, 255, 255, 0.12)",
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "right",
                    labels: { color: "#8fa0c0", font: { family: "Outfit", size: 12 } }
                }
            }
        }
    });
}

function renderDpiMlChart(dpiCount, mlCount) {
    const canvas = document.getElementById("dpiMlChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (dpiMlChartInstance) {
        dpiMlChartInstance.destroy();
    }

    dpiMlChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Rule-Based DPI", "ML Behavioral Fallback"],
            datasets: [{
                label: "Classified Flows",
                data: [dpiCount, mlCount],
                backgroundColor: [
                    "rgba(0, 242, 254, 0.75)",
                    "rgba(127, 0, 255, 0.75)"
                ],
                borderColor: [
                    "#00f2fe",
                    "#7f00ff"
                ],
                borderWidth: 1,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: "#8fa0c0" }, grid: { display: false } },
                y: { ticks: { color: "#8fa0c0" }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
            }
        }
    });
}

function renderThroughputChart(throughputBuckets) {
    const canvas = document.getElementById("throughputChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (throughputChartInstance) {
        throughputChartInstance.destroy();
    }

    const labels = Object.keys(throughputBuckets);
    const counts = Object.values(throughputBuckets);

    throughputChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Flows",
                data: counts,
                backgroundColor: "rgba(79, 172, 254, 0.7)",
                borderColor: "#4facfe",
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: "#8fa0c0", font: { size: 11 } }, grid: { display: false } },
                y: { ticks: { color: "#8fa0c0" }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
            }
        }
    });
}

function renderFeatureChart(featureList) {
    const canvas = document.getElementById("featureChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    if (featureChartInstance) {
        featureChartInstance.destroy();
    }

    const labels = featureList.map(f => f.name);
    const dataValues = featureList.map(f => f.percent);

    featureChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Feature Importance (%)",
                data: dataValues,
                backgroundColor: "rgba(0, 230, 118, 0.7)",
                borderColor: "#00e676",
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: "#8fa0c0", callback: v => `${v}%` },
                    grid: { color: "rgba(255, 255, 255, 0.05)" }
                },
                y: { ticks: { color: "#8fa0c0", font: { size: 11 } }, grid: { display: false } }
            }
        }
    });
}

// ==============================================================================
// 4. DATASET TABLE RENDERING & FILTERING
// ==============================================================================
function renderDatasetTable(flows) {
    const tbody = document.getElementById("dataset-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (!flows || flows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No matching flows found.</td></tr>`;
        return;
    }

    flows.forEach(flow => {
        const tr = document.createElement("tr");
        const badgeClass = flow.method === "DPI" ? "badge-dpi" : "badge-ml";
        tr.innerHTML = `
            <td><span class="badge ${badgeClass}">${flow.method}</span></td>
            <td><strong>${flow.app}</strong></td>
            <td>${flow.duration_s}s</td>
            <td>${Number(flow.packets).toLocaleString()}</td>
            <td>${Number(flow.bytes_kb).toLocaleString()} KB</td>
            <td style="color: var(--text-muted); font-family: monospace; font-size: 0.8rem;">${flow.sni}</td>
            <td><span style="color: #4facfe;">${flow.dpi_type}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function filterDatasetTable() {
    const input = document.getElementById("flow-search");
    if (!input) return;
    const term = input.value.toLowerCase().trim();
    if (!term) {
        renderDatasetTable(fullDatasetCache);
        return;
    }
    const filtered = fullDatasetCache.filter(f => 
        (f.app && f.app.toLowerCase().includes(term)) ||
        (f.sni && f.sni.toLowerCase().includes(term)) ||
        (f.method && f.method.toLowerCase().includes(term)) ||
        (f.dpi_type && f.dpi_type.toLowerCase().includes(term))
    );
    renderDatasetTable(filtered);
}
window.filterDatasetTable = filterDatasetTable;

// ==============================================================================
// 5. FLOW CALCULATOR PRESETS & INFERENCE
// ==============================================================================
function applyPreset(type) {
    const p = presets[type];
    if (!p) return;
    const durInput = document.getElementById("duration");
    const pktInput = document.getElementById("packets");
    const bytesInput = document.getElementById("bytes");
    const meanInput = document.getElementById("mean-size");
    const iatInput = document.getElementById("iat");

    if (durInput) durInput.value = p.duration;
    if (pktInput) pktInput.value = p.packets;
    if (bytesInput) bytesInput.value = p.bytes;
    if (meanInput) meanInput.value = p.mean;
    if (iatInput) iatInput.value = p.iat;
}
window.applyPreset = applyPreset;

// ==============================================================================
// 6. HEALTH CHECK
// ==============================================================================
async function checkBackendHealth() {
    const statusDot = document.getElementById("status-dot");
    const statusLabel = document.getElementById("status-label");
    if (!statusDot || !statusLabel) return;

    try {
        const res = await fetch("/health");
        if (res.ok) {
            const data = await res.json();
            if (data.status === "healthy" && data.model_loaded) {
                statusDot.className = "status-dot green";
                statusLabel.innerHTML = "ML Service: <strong>Online</strong>";
                return;
            }
        }
    } catch (e) {
        // Backend health error handled below
    }
    statusDot.className = "status-dot yellow";
    statusLabel.innerHTML = "ML Service: <strong>Offline</strong>";
}

// ==============================================================================
// 7. INITIALIZATION ON DOMContentLoaded
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
    // 1. Initial health ping
    checkBackendHealth();

    // 2. Setup Form Submission for Inference
    const form = document.getElementById("inference-form");
    const resultBox = document.getElementById("prediction-result");
    const predictedApp = document.getElementById("predicted-app");
    const predictedConfBar = document.getElementById("predicted-conf-bar");
    const predictedConfText = document.getElementById("predicted-conf-text");
    const anomalyBadge = document.getElementById("anomaly-badge");
    const historyBody = document.getElementById("history-table-body");
    const classifyBtn = document.getElementById("btn-classify");

    if (form) {
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

            if (classifyBtn) {
                classifyBtn.textContent = "Calculating Inference...";
                classifyBtn.disabled = true;
            }

            try {
                const res = await fetch("/predict-flow", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) throw new Error("Inference error: " + res.status);
                const data = await res.json();

                const appName = String(data.prediction).replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
                const confPct = Math.round(data.confidence * 100);

                if (predictedApp) predictedApp.textContent = appName;
                if (predictedConfBar) predictedConfBar.style.width = `${confPct}%`;
                if (predictedConfText) predictedConfText.textContent = `${confPct}% Confidence`;

                if (anomalyBadge) {
                    if (data.is_anomaly) {
                        anomalyBadge.className = "badge badge-anomaly";
                        anomalyBadge.textContent = "⚠️ Anomaly Detected";
                    } else {
                        anomalyBadge.className = "badge badge-normal";
                        anomalyBadge.textContent = "✓ Normal Pattern";
                    }
                }

                if (resultBox) resultBox.classList.remove("hidden");

                if (historyBody) {
                    const newRow = document.createElement("tr");
                    newRow.innerHTML = `
                        <td><span class="badge badge-ml">ML</span></td>
                        <td><strong>${appName}</strong></td>
                        <td>${(duration / 1000).toFixed(1)}s</td>
                        <td>${(bytes / 1024).toFixed(1)} KB</td>
                        <td>${confPct}%</td>
                        <td>${data.is_anomaly ? '<span class="badge badge-anomaly">Anomaly</span>' : '<span class="badge badge-normal">Normal</span>'}</td>
                    `;
                    historyBody.insertBefore(newRow, historyBody.firstChild);
                }

            } catch (err) {
                console.error("Inference request failed:", err);
                alert("Could not reach ML inference backend. Make sure the server is running.");
            } finally {
                if (classifyBtn) {
                    classifyBtn.textContent = "Classify Traffic Flow";
                    classifyBtn.disabled = false;
                }
            }
        });
    }

    // 3. Determine initial view from hash or pathname
    const currentHash = window.location.hash.toLowerCase();
    const currentPath = window.location.pathname.toLowerCase();
    if (currentHash === "#calculate" || currentHash === "#calc" || currentPath.includes("calc")) {
        switchView("calc");
    } else {
        switchView("data");
    }
});

// 4. Handle browser Back/Forward hash changes dynamically
window.addEventListener("hashchange", () => {
    const currentHash = window.location.hash.toLowerCase();
    if (currentHash === "#calculate" || currentHash === "#calc") {
        switchView("calc");
    } else {
        switchView("data");
    }
});
