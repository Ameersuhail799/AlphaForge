/* AlphaForge Dashboard Application Logic & REST Integration */

let currentAsset = "tcs_ns";
let livePollInterval = null;
let tvChartInstance = null;
let marketChart = null;
let currentChartType = "area"; // "area" | "candlestick"
let currentChartTF = "3M";     // "1D" | "1W" | "1M" | "3M" | "6M" | "1Y" | "ALL"
let showSMAOverlay = true;
let showVolOverlay = true;
let currentRawMarketData = null;

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    setupTabNavigation();
    setupEventListeners();
    setupChartControlListeners();
    loadSupportedAssets();
    refreshCurrentAssetView(currentAsset);
    refreshPortfolioView();
    loadBacktestResearchSummaries();
    initStrategyTesterUI();

    if (livePollInterval) clearInterval(livePollInterval);
    livePollInterval = setInterval(() => {
        loadLivePrice(currentAsset);
    }, 30000);
}

function setupChartControlListeners() {
    const typePills = document.querySelectorAll("#chart-type-pills .pill-btn");
    typePills.forEach(btn => {
        btn.addEventListener("click", () => {
            typePills.forEach(b => {
                b.classList.remove("active");
                b.style.background = "transparent";
                b.style.color = "#9CA3AF";
            });
            btn.classList.add("active");
            btn.style.background = "#6366F1";
            btn.style.color = "#FFF";
            currentChartType = btn.getAttribute("data-chart-type");
            if (currentRawMarketData) renderChartWithFailsafe(currentRawMarketData);
        });
    });

    const tfPills = document.querySelectorAll("#chart-tf-pills .pill-btn");
    tfPills.forEach(btn => {
        btn.addEventListener("click", () => {
            tfPills.forEach(b => {
                b.classList.remove("active");
                b.style.background = "transparent";
                b.style.color = "#9CA3AF";
            });
            btn.classList.add("active");
            btn.style.background = "#6366F1";
            btn.style.color = "#FFF";
            currentChartTF = btn.getAttribute("data-tf");
            if (currentRawMarketData) renderChartWithFailsafe(currentRawMarketData);
        });
    });

    const smaBtn = document.getElementById("btn-toggle-sma");
    if (smaBtn) {
        smaBtn.addEventListener("click", () => {
            showSMAOverlay = !showSMAOverlay;
            smaBtn.style.background = showSMAOverlay ? "#6366F1" : "transparent";
            smaBtn.style.color = showSMAOverlay ? "#FFF" : "#9CA3AF";
            if (currentRawMarketData) renderChartWithFailsafe(currentRawMarketData);
        });
    }

    const volBtn = document.getElementById("btn-toggle-vol");
    if (volBtn) {
        volBtn.addEventListener("click", () => {
            showVolOverlay = !showVolOverlay;
            volBtn.style.background = showVolOverlay ? "#6366F1" : "transparent";
            volBtn.style.color = showVolOverlay ? "#FFF" : "#9CA3AF";
            if (currentRawMarketData) renderChartWithFailsafe(currentRawMarketData);
        });
    }
}

function setupTabNavigation() {
    const navButtons = document.querySelectorAll(".side-nav .nav-btn");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const tabId = btn.getAttribute("data-tab");
            document.querySelectorAll(".tab-page").forEach(page => page.classList.remove("active"));
            const targetPage = document.getElementById(tabId);
            if (targetPage) targetPage.classList.add("active");
        });
    });
}

function setupEventListeners() {
    document.getElementById("btn-execute-buy").addEventListener("click", () => handlePaperTrade("BUY"));
    document.getElementById("btn-execute-sell").addEventListener("click", () => handlePaperTrade("SELL"));
    document.getElementById("btn-reset-portfolio").addEventListener("click", handleResetPortfolio);

    const capInput = document.getElementById("trade-capital-input");
    if (capInput) {
        capInput.addEventListener("input", () => {
            capInput.dataset.userEdited = "true";
        });
    }
}

async function loadSupportedAssets() {
    try {
        const res = await fetch("/api/assets");
        const data = await res.json();
        const container = document.getElementById("asset-tabs-container");
        container.innerHTML = "";

        data.assets.forEach(asset => {
            const btn = document.createElement("button");
            btn.className = `asset-tab-btn ${asset.symbol === currentAsset ? 'active' : ''}`;
            btn.setAttribute("data-symbol", asset.symbol);
            btn.innerText = `${asset.display_name.split(' ')[0]} (₹${asset.last_price.toFixed(1)})`;
            btn.onclick = () => switchAsset(asset.symbol);
            container.appendChild(btn);
        });
    } catch (err) {
        console.error("Failed to load assets:", err);
    }
}

function switchAsset(symbol) {
    currentAsset = symbol;
    const buttons = document.querySelectorAll(".asset-tab-btn");
    buttons.forEach(b => {
        const btnSym = b.getAttribute("data-symbol");
        const match = (btnSym && btnSym.toLowerCase() === symbol.toLowerCase()) || 
                      b.innerText.toLowerCase().startsWith(symbol.split('_')[0].toLowerCase());
        if (match) {
            b.classList.add("active");
        } else {
            b.classList.remove("active");
        }
    });
    refreshCurrentAssetView(symbol);
}

window.switchAsset = switchAsset;

function refreshCurrentAssetView(symbol) {
    loadChartData(symbol);
    loadSignalData(symbol);
    loadLivePrice(symbol);
    loadHistoricalContext(symbol);
}

async function loadHistoricalContext(symbol) {
    try {
        const res = await fetch(`/api/historical-context?symbol=${symbol}`);
        const data = await res.json();

        const regimeDescEl = document.getElementById("hc-regime-desc");
        const occTextEl = document.getElementById("hc-occurrences-text");
        const insufficientBox = document.getElementById("hc-insufficient-sample-box");
        const distContainer = document.getElementById("hc-distribution-container");

        if (regimeDescEl) regimeDescEl.innerText = data.regime_description || "Unknown Market Regime";
        if (occTextEl) occTextEl.innerText = data.message || "";

        if (data.insufficient_sample) {
            if (insufficientBox) {
                insufficientBox.innerText = `⚠️ Not enough similar days in history (N=${data.sample_count}) to draw a pattern.`;
                insufficientBox.style.display = "block";
            }
            if (distContainer) distContainer.style.display = "none";
        } else {
            if (insufficientBox) insufficientBox.style.display = "none";
            if (distContainer) distContainer.style.display = "flex";

            const m = data.metrics;
            const posBadge = document.getElementById("hc-pos-pct-badge");
            const minEl = document.getElementById("hc-stat-min");
            const p25El = document.getElementById("hc-stat-p25");
            const medEl = document.getElementById("hc-stat-median");
            const p75El = document.getElementById("hc-stat-p75");
            const maxEl = document.getElementById("hc-stat-max");

            if (posBadge) {
                posBadge.innerText = `+${m.pct_positive.toFixed(2)}% Positive`;
                posBadge.style.color = m.pct_positive >= 50 ? "#34D399" : "#F87171";
            }

            if (minEl) minEl.innerText = `${m.min_ret_pct >= 0 ? '+' : ''}${m.min_ret_pct.toFixed(2)}%`;
            if (p25El) p25El.innerText = `${m.p25_ret_pct >= 0 ? '+' : ''}${m.p25_ret_pct.toFixed(2)}%`;
            if (medEl) medEl.innerText = `${m.median_ret_pct >= 0 ? '+' : ''}${m.median_ret_pct.toFixed(2)}%`;
            if (p75El) p75El.innerText = `${m.p75_ret_pct >= 0 ? '+' : ''}${m.p75_ret_pct.toFixed(2)}%`;
            if (maxEl) maxEl.innerText = `${m.max_ret_pct >= 0 ? '+' : ''}${m.max_ret_pct.toFixed(2)}%`;
        }
    } catch (err) {
        console.error("Failed to load historical context:", err);
    }
}

async function loadLivePrice(symbol) {
    try {
        const res = await fetch(`/api/live-price?symbol=${symbol}`);
        const data = await res.json();

        const symUpper = symbol.replace("_", ".").toUpperCase();
        const labelEl = document.getElementById("nav-live-symbol-label");
        if (labelEl) labelEl.innerText = `${symUpper.split('.')[0]} LIVE TICK`;

        const priceFormatted = `₹${data.current_price.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

        const priceEl = document.getElementById("nav-live-price");
        if (priceEl) {
            const sign = data.change_val >= 0 ? "+" : "";
            priceEl.innerText = `${priceFormatted} (${sign}${data.change_pct.toFixed(2)}%)`;
            priceEl.className = `chip-value ${data.change_val >= 0 ? 'positive' : 'negative'}`;
        }

        // TASK 2: Wire signal-last-price card display to the exact same live tick source!
        const sigPriceEl = document.getElementById("signal-last-price");
        if (sigPriceEl) {
            sigPriceEl.innerText = priceFormatted;
        }

        const marketText = document.getElementById("market-status-text");
        if (marketText) marketText.innerText = data.market_status_text;

        const marketBadge = document.getElementById("signal-market-status");
        if (marketBadge) {
            marketBadge.innerText = data.is_market_open ? "🟢 Market Open" : "🔴 Market Closed";
            marketBadge.style.color = data.is_market_open ? "#10B981" : "#9CA3AF";
            marketBadge.style.backgroundColor = data.is_market_open ? "rgba(16, 185, 129, 0.15)" : "rgba(255, 255, 255, 0.08)";
        }
    } catch (err) {
        console.error("Failed to load live price:", err);
    }
}

async function loadSignalData(symbol) {
    try {
        const res = await fetch(`/api/signal?symbol=${symbol}`);
        const data = await res.json();

        document.getElementById("signal-asset-name").innerText = data.display_name;
        document.getElementById("signal-asset-symbol").innerText = data.symbol.toUpperCase();

        const asOfEl = document.getElementById("signal-as-of-date");
        if (asOfEl) {
            asOfEl.innerText = data.signal_timestamp_text || `Confirmed Daily Close: ${data.timestamp}`;
        }

        const badge = document.getElementById("signal-action-badge");
        badge.innerText = data.signal;
        badge.style.backgroundColor = data.signal_color;
        badge.style.color = "#FFFFFF";

        document.getElementById("signal-prob-pct").innerText = `${data.prob_up_pct.toFixed(1)}%`;
        document.getElementById("signal-exp-ret").innerText = `${data.expected_return_pct >= 0 ? '+' : ''}${data.expected_return_pct.toFixed(2)}%`;
        document.getElementById("signal-risk-level").innerText = data.risk_level;

        document.getElementById("prob-bar-label").innerText = `${data.prob_up_pct.toFixed(1)}%`;
        document.getElementById("prob-bar-fill").style.width = `${Math.min(data.prob_up_pct, 100)}%`;

        const retWidth = Math.min(Math.max((data.expected_return_pct / 4.0) * 100, 5), 100);
        document.getElementById("ret-bar-label").innerText = `${data.expected_return_pct >= 0 ? '+' : ''}${data.expected_return_pct.toFixed(2)}%`;
        document.getElementById("ret-bar-fill").style.width = `${retWidth}%`;

        const evidenceList = document.getElementById("evidence-reasons-list");
        evidenceList.innerHTML = "";
        data.reasons.forEach(reason => {
            const li = document.createElement("li");
            li.innerText = reason;
            evidenceList.appendChild(li);
        });
    } catch (err) {
        console.error("Failed to load signal data:", err);
    }
}

async function loadChartData(symbol) {
    try {
        console.log(`[loadChartData] Fetching /api/market-data?symbol=${symbol}...`);
        const res = await fetch(`/api/market-data?symbol=${symbol}`);
        if (!res.ok) {
            throw new Error(`HTTP Error ${res.status}: ${res.statusText}`);
        }
        const data = await res.json();
        console.log(`[loadChartData] Response received. Total rows: ${data.close ? data.close.length : 0}`);
        currentRawMarketData = data;
        renderChartWithFailsafe(data);
    } catch (err) {
        console.error("[loadChartData] Critical error fetching chart data:", err);
    }
}

function renderChartWithFailsafe(data) {
    try {
        renderLightweightChart(data);
    } catch (tvErr) {
        console.warn("[renderChartWithFailsafe] LightweightCharts renderer threw an exception, executing Chart.js fallback:", tvErr);
        renderPlainChartJSFallback(data);
    }
}

function createTVSeries(chart, seriesType, options) {
    if (typeof chart.addSeries === "function" && LightweightCharts[seriesType]) {
        return chart.addSeries(LightweightCharts[seriesType], options);
    }
    const methodName = "add" + seriesType;
    if (typeof chart[methodName] === "function") {
        return chart[methodName](options);
    }
    throw new Error(`Neither addSeries nor ${methodName} is supported.`);
}

function renderLightweightChart(data) {
    if (typeof LightweightCharts === "undefined") {
        throw new Error("TradingView LightweightCharts library is not loaded.");
    }

    const container = document.getElementById("lightweight-chart-container");
    if (!container) throw new Error("Chart container element 'lightweight-chart-container' not found.");

    const totalBars = data.dates.length;
    let barCount = totalBars;

    if (currentChartTF === "1D") barCount = 1;
    else if (currentChartTF === "1W") barCount = 5;
    else if (currentChartTF === "1M") barCount = 22;
    else if (currentChartTF === "3M") barCount = 66;
    else if (currentChartTF === "6M") barCount = 132;
    else if (currentChartTF === "1Y") barCount = 252;
    else if (currentChartTF === "ALL") barCount = totalBars;

    const startIdx = Math.max(0, totalBars - barCount);

    let rawDates = data.dates.slice(startIdx);
    let rawOpen = (data.open && data.open.length === totalBars ? data.open : data.close).slice(startIdx);
    let rawHigh = (data.high && data.high.length === totalBars ? data.high : data.close).slice(startIdx);
    let rawLow = (data.low && data.low.length === totalBars ? data.low : data.close).slice(startIdx);
    let rawClose = data.close.slice(startIdx);
    let rawVol = (data.volume && data.volume.length === totalBars ? data.volume : rawDates.map(() => 1000)).slice(startIdx);
    let rawSma20 = data.sma20 ? data.sma20.slice(startIdx) : [];
    let rawSma50 = data.sma50 ? data.sma50.slice(startIdx) : [];

    // Forming live price bar (only push if today date is strictly greater than latest historical close)
    let isLiveActive = false;
    if (data.live_price_info && data.live_price_info.current_price) {
        const liveP = data.live_price_info.current_price;
        const prevC = data.live_price_info.previous_close || rawClose[rawClose.length - 1];
        const now = new Date();
        const y = now.getFullYear();
        const m = String(now.getMonth() + 1).padStart(2, '0');
        const d = String(now.getDate()).padStart(2, '0');
        const todayStr = `${y}-${m}-${d}`;

        const lastHistoricalDate = rawDates[rawDates.length - 1];
        if (lastHistoricalDate && todayStr > lastHistoricalDate) {
            rawDates.push(todayStr);
            rawOpen.push(prevC);
            rawHigh.push(Math.max(liveP, prevC));
            rawLow.push(Math.min(liveP, prevC));
            rawClose.push(liveP);
            rawVol.push(rawVol.length > 0 ? rawVol[rawVol.length - 1] : 1000);
            if (rawSma20.length > 0) rawSma20.push(rawSma20[rawSma20.length - 1]);
            if (rawSma50.length > 0) rawSma50.push(rawSma50[rawSma50.length - 1]);
            isLiveActive = true;
        }
    }

    const firstClose = rawClose[0] || 1.0;
    const latestClose = rawClose[rawClose.length - 1] || 1.0;
    const tfReturnPct = ((latestClose - firstClose) / firstClose) * 100.0;
    const tfSign = tfReturnPct >= 0 ? "+" : "";

    const nameEl = document.getElementById("chart-asset-display-name");
    if (nameEl) nameEl.innerText = `${data.display_name} Chart`;

    const badgeEl = document.getElementById("chart-tf-return-badge");
    if (badgeEl) {
        badgeEl.innerText = `${tfSign}${tfReturnPct.toFixed(2)}% (${currentChartTF})`;
        badgeEl.style.color = tfReturnPct >= 0 ? "#10B981" : "#EF4444";
        badgeEl.style.backgroundColor = tfReturnPct >= 0 ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)";
    }

    const lastIdx = rawClose.length - 1;
    updateTVHeader(rawDates[lastIdx], rawOpen[lastIdx], rawHigh[lastIdx], rawLow[lastIdx], rawClose[lastIdx], rawVol[lastIdx]);

    container.innerHTML = `<div id="tv-chart-wrapper" style="width:100%; height:380px;"></div>`;
    const wrapper = document.getElementById("tv-chart-wrapper");

    if (tvChartInstance) {
        try { tvChartInstance.remove(); } catch(e) {}
        tvChartInstance = null;
    }

    const initialWidth = Math.max(wrapper.clientWidth || container.clientWidth || 0, 500);

    tvChartInstance = LightweightCharts.createChart(wrapper, {
        width: initialWidth,
        height: 380,
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#9CA3AF',
            fontSize: 11,
        },
        grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
        },
        timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
            timeVisible: true,
        },
    });

    const isUp = tfReturnPct >= 0;
    const mainColor = isUp ? "#10B981" : "#EF4444";

    if (currentChartType === "area") {
        const areaSeries = createTVSeries(tvChartInstance, "AreaSeries", {
            topColor: isUp ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)',
            bottomColor: isUp ? 'rgba(16, 185, 129, 0.0)' : 'rgba(239, 68, 68, 0.0)',
            lineColor: mainColor,
            lineWidth: 2,
        });
        const areaData = rawDates.map((t, i) => ({ time: t, value: rawClose[i] }));
        areaSeries.setData(areaData);
    } else {
        const candleSeries = createTVSeries(tvChartInstance, "CandlestickSeries", {
            upColor: '#10B981',
            downColor: '#EF4444',
            borderVisible: false,
            wickUpColor: '#10B981',
            wickDownColor: '#EF4444',
        });
        const candleData = rawDates.map((t, i) => ({
            time: t,
            open: rawOpen[i],
            high: rawHigh[i],
            low: rawLow[i],
            close: rawClose[i],
        }));
        candleSeries.setData(candleData);
    }

    // Muted Volume Histogram
    const volSeries = createTVSeries(tvChartInstance, "HistogramSeries", {
        priceFormat: { type: 'volume' },
        priceScaleId: '',
        scaleMargins: { top: 0.8, bottom: 0 },
    });
    const volData = rawDates.map((t, i) => {
        const o = rawOpen[i] || rawClose[i];
        const c = rawClose[i];
        return {
            time: t,
            value: rawVol[i],
            color: c >= o ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)',
        };
    });
    volSeries.setData(volData);

    // SMA Overlays
    if (showSMAOverlay && rawSma20.length > 0) {
        const sma20Series = createTVSeries(tvChartInstance, "LineSeries", {
            color: '#6366F1',
            lineWidth: 1,
            lineStyle: 2,
        });
        sma20Series.setData(rawDates.map((t, i) => ({ time: t, value: rawSma20[i] })));
    }

    if (showSMAOverlay && rawSma50.length > 0) {
        const sma50Series = createTVSeries(tvChartInstance, "LineSeries", {
            color: '#F59E0B',
            lineWidth: 1,
            lineStyle: 3,
        });
        sma50Series.setData(rawDates.map((t, i) => ({ time: t, value: rawSma50[i] })));
    }

    // Crosshair Hover Readout
    tvChartInstance.subscribeCrosshairMove((param) => {
        if (!param || !param.time || param.point === undefined || param.point.x < 0 || param.point.y < 0) {
            updateTVHeader(rawDates[lastIdx], rawOpen[lastIdx], rawHigh[lastIdx], rawLow[lastIdx], rawClose[lastIdx], rawVol[lastIdx]);
            return;
        }
        let timeStr = typeof param.time === 'string' ? param.time : '';
        if (param.time && typeof param.time === 'object' && param.time.year) {
            const mStr = String(param.time.month).padStart(2, '0');
            const dStr = String(param.time.day).padStart(2, '0');
            timeStr = `${param.time.year}-${mStr}-${dStr}`;
        }
        const matchIdx = rawDates.indexOf(timeStr);
        if (matchIdx !== -1) {
            updateTVHeader(timeStr, rawOpen[matchIdx], rawHigh[matchIdx], rawLow[matchIdx], rawClose[matchIdx], rawVol[matchIdx]);
        }
    });

    if (window.ResizeObserver && wrapper) {
        const ro = new ResizeObserver(entries => {
            if (!entries || !entries.length) return;
            const w = Math.max(entries[0].contentRect.width, 300);
            if (tvChartInstance && w > 0) {
                tvChartInstance.applyOptions({ width: w });
            }
        });
        ro.observe(wrapper);
    }

    tvChartInstance.timeScale().fitContent();
    console.log("[renderLightweightChart] TradingView Lightweight Chart rendered successfully with 0 errors!");
}

function updateTVHeader(d, o, h, l, c, v) {
    const dEl = document.getElementById("tv-date");
    const oEl = document.getElementById("tv-open");
    const hEl = document.getElementById("tv-high");
    const lEl = document.getElementById("tv-low");
    const cEl = document.getElementById("tv-close");
    const vEl = document.getElementById("tv-vol");

    if (dEl) dEl.innerText = d || "-";
    if (oEl) oEl.innerText = o ? `₹${o.toFixed(2)}` : "-";
    if (hEl) hEl.innerText = h ? `₹${h.toFixed(2)}` : "-";
    if (lEl) lEl.innerText = l ? `₹${l.toFixed(2)}` : "-";
    if (cEl) cEl.innerText = c ? `₹${c.toFixed(2)}` : "-";
    if (vEl) vEl.innerText = v ? (v >= 1e6 ? `${(v/1e6).toFixed(2)}M` : `${(v/1e3).toFixed(1)}K`) : "-";
}

function renderPlainChartJSFallback(data) {
    console.log("[renderPlainChartJSFallback] Executing plain Chart.js fallback...");
    const nameEl = document.getElementById("chart-asset-display-name");
    if (nameEl && data.display_name) nameEl.innerText = `${data.display_name} Chart`;

    const container = document.getElementById("lightweight-chart-container");
    if (container) {
        container.innerHTML = `<div style="position:relative; width:100%; height:380px;"><canvas id="market-chart-canvas"></canvas></div>`;
    }

    const canvasEl = document.getElementById("market-chart-canvas");
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (marketChart) marketChart.destroy();

    let labels = [...data.dates];
    let prices = [...data.close];

    marketChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price (₹)',
                data: prices,
                borderColor: '#06B6D4',
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9CA3AF' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9CA3AF' } }
            }
        }
    });
}

async function handlePaperTrade(action) {
    const capitalInput = parseFloat(document.getElementById("trade-capital-input").value) || 20000;
    try {
        const res = await fetch("/api/paper-trade", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol: currentAsset, action: action, capital: capitalInput })
        });
        const result = await res.json();

        if (result.status === "SUCCESS") {
            alert(`Paper ${action} Trade Executed Successfully!`);
            await refreshPortfolioView();
        } else {
            alert(`Trade Failed: ${result.message}`);
        }
    } catch (err) {
        alert("Error executing paper trade.");
    }
}

async function handleResetPortfolio() {
    if (!confirm("Are you sure you want to reset paper portfolio to ₹1,00,000?")) return;
    try {
        await fetch("/api/portfolio/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ initial_capital: 100000 })
        });
        await refreshPortfolioView();
    } catch (err) {
        alert("Error resetting portfolio.");
    }
}

function switchToTab(tabId) {
    const navButtons = document.querySelectorAll(".side-nav .nav-btn");
    navButtons.forEach(btn => {
        if (btn.getAttribute("data-tab") === tabId) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });
    document.querySelectorAll(".tab-page").forEach(page => page.classList.remove("active"));
    const targetPage = document.getElementById(tabId);
    if (targetPage) targetPage.classList.add("active");
}

async function refreshPortfolioView() {
    try {
        const res = await fetch("/api/portfolio");
        const data = await res.json();

        // Dynamically compute 20% portfolio equity cap for paper trade capital input
        const max20Cap = data.current_equity * 0.20;
        const capInput = document.getElementById("trade-capital-input");
        if (capInput) {
            capInput.max = Math.floor(max20Cap);
            const currVal = parseFloat(capInput.value);
            if (isNaN(currVal) || currVal > max20Cap || !capInput.dataset.userEdited) {
                capInput.value = Math.floor(max20Cap);
            }
        }
        const capLabel = document.getElementById("trade-capital-label");
        if (capLabel) {
            capLabel.innerText = `Capital Allocation (Max 20% Equity: ₹${max20Cap.toLocaleString('en-IN', {maximumFractionDigits: 0})}):`;
        }

        document.getElementById("nav-portfolio-equity").innerText = `₹${data.current_equity.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        document.getElementById("nav-cash-balance").innerText = `₹${data.cash_balance.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

        const pnlEl = document.getElementById("nav-total-pnl");
        const sign = data.total_pnl >= 0 ? "+" : "";
        pnlEl.innerText = `${sign}₹${data.total_pnl.toLocaleString('en-IN', {minimumFractionDigits: 2})} (${sign}${data.total_pnl_pct.toFixed(2)}%)`;
        pnlEl.className = `chip-value ${data.total_pnl >= 0 ? 'positive' : 'negative'}`;

        document.getElementById("nav-win-rate").innerText = `${data.win_rate_pct.toFixed(1)}%`;

        document.getElementById("port-total-equity").innerText = `₹${data.current_equity.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        document.getElementById("port-cash").innerText = `₹${data.cash_balance.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        document.getElementById("port-realized-pnl").innerText = `₹${data.realized_pnl.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        document.getElementById("port-unrealized-pnl").innerText = `₹${data.unrealized_pnl.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

        const openTbody = document.getElementById("open-positions-tbody");
        openTbody.innerHTML = "";
        if (data.open_positions.length === 0) {
            openTbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: #9CA3AF;">No open positions. Select an asset and click 'BUY POSITION'.</td></tr>`;
        } else {
            data.open_positions.forEach(pos => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${pos.asset.toUpperCase()}</strong></td>
                    <td>${pos.entry_date}</td>
                    <td>₹${pos.entry_price.toFixed(2)}</td>
                    <td>₹${pos.current_price.toFixed(2)}</td>
                    <td>${pos.units.toFixed(2)}</td>
                    <td>₹${pos.allocated_cash.toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
                    <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">₹${pos.unrealized_pnl.toFixed(2)} (${pos.unrealized_pnl_pct.toFixed(2)}%)</td>
                    <td><button class="btn btn-sell" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;" onclick="closePositionDirect('${pos.asset}')">CLOSE</button></td>
                `;
                openTbody.appendChild(tr);
            });
        }

        const histTbody = document.getElementById("trade-history-tbody");
        histTbody.innerHTML = "";
        if (data.trade_history.length === 0) {
            histTbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: #9CA3AF;">No closed trade history.</td></tr>`;
        } else {
            data.trade_history.forEach(trRecord => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>#${trRecord.trade_id}</td>
                    <td><strong>${trRecord.asset.toUpperCase()}</strong></td>
                    <td>${trRecord.entry_date}</td>
                    <td>${trRecord.exit_date}</td>
                    <td>₹${trRecord.entry_price.toFixed(2)}</td>
                    <td>₹${trRecord.exit_price.toFixed(2)}</td>
                    <td>${trRecord.units.toFixed(2)}</td>
                    <td class="${trRecord.net_pnl >= 0 ? 'positive' : 'negative'}">₹${trRecord.net_pnl.toFixed(2)}</td>
                    <td class="${trRecord.net_return_pct >= 0 ? 'positive' : 'negative'}">${trRecord.net_return_pct >= 0 ? '+' : ''}${trRecord.net_return_pct.toFixed(2)}%</td>
                `;
                histTbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Failed to refresh portfolio view:", err);
    }
}

async function closePositionDirect(symbol) {
    try {
        const res = await fetch("/api/paper-trade", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol: symbol, action: "CLOSE" })
        });
        const result = await res.json();
        if (result.status === "SUCCESS") {
            await refreshPortfolioView();
        } else {
            alert(`Close Failed: ${result.message}`);
        }
    } catch (err) {
        alert("Error closing position.");
    }
}

async function loadBacktestResearchSummaries() {
    try {
        const res = await fetch("/api/backtests");
        const data = await res.json();
        
        // 1. Render Realistic Per-Stock Benchmark Table
        const realityTbody = document.getElementById("reality-check-tbody");
        if (realityTbody && data.per_stock_reality_check) {
            realityTbody.innerHTML = "";
            data.per_stock_reality_check.forEach(row => {
                const tr = document.createElement("tr");
                const diffClass = row.cagr_diff >= 0 ? "positive" : "negative";
                tr.innerHTML = `
                    <td><strong>${row.asset.toUpperCase()}</strong></td>
                    <td>${row.champion_cagr.toFixed(2)}%</td>
                    <td>${row.bh_cagr.toFixed(2)}%</td>
                    <td class="${diffClass}">${row.cagr_diff >= 0 ? '+' : ''}${row.cagr_diff.toFixed(2)}%</td>
                    <td>${row.champion_sharpe.toFixed(2)}</td>
                    <td>${row.bh_sharpe.toFixed(2)}</td>
                    <td><span style="color: #EF4444; font-weight: 700;">${row.verdict}</span></td>
                `;
                realityTbody.appendChild(tr);
            });
        }

        // 2. Render Superseded Historical Matrix Table
        const tbody = document.getElementById("backtest-summary-tbody");
        if (tbody && data.superseded_historical_matrix) {
            tbody.innerHTML = "";
            data.superseded_historical_matrix.forEach(row => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>${row.asset.toUpperCase()}</strong></td>
                    <td>${row.display_name}</td>
                    <td class="positive">+${row.cum_return_pct.toFixed(2)}%</td>
                    <td>${row.sharpe.toFixed(2)}</td>
                    <td>+${row.expectancy_pct.toFixed(2)}% / trade</td>
                    <td><span style="color: #9CA3AF; font-weight: 600;">${row.positive_folds}</span></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Failed to load backtest summaries:", err);
    }
}

/* Strategy Tester UI Controller */
function initStrategyTesterUI() {
    const templateSelect = document.getElementById("st-template-select");
    const runBtn = document.getElementById("btn-run-strategy-test");

    if (templateSelect) {
        templateSelect.addEventListener("change", renderStrategyTesterParams);
        renderStrategyTesterParams();
    }

    if (runBtn) {
        runBtn.addEventListener("click", handleRunStrategyTest);
    }
}

function renderStrategyTesterParams() {
    const templateSelect = document.getElementById("st-template-select");
    if (!templateSelect) return;
    const template = templateSelect.value;
    const container = document.getElementById("st-params-container");
    if (!container) return;

    if (template === "rsi_threshold") {
        container.innerHTML = `
            <div class="form-group">
                <label style="font-size: 0.82rem; color: #D1D5DB;">RSI Period:</label>
                <input type="number" id="st-param-period" value="14" min="2" max="100" style="width:100%; padding:0.5rem; background:rgba(15,23,42,0.8); border:1px solid var(--border-color); border-radius:6px; color:#FFF; margin-top:0.25rem;">
            </div>
            <div class="form-group">
                <label style="font-size: 0.82rem; color: #D1D5DB;">Buy Threshold (Cross Below):</label>
                <input type="number" id="st-param-buy-threshold" value="30" min="5" max="95" style="width:100%; padding:0.5rem; background:rgba(15,23,42,0.8); border:1px solid var(--border-color); border-radius:6px; color:#FFF; margin-top:0.25rem;">
            </div>
        `;
    } else if (template === "sma_crossover") {
        container.innerHTML = `
            <div class="form-group">
                <label style="font-size: 0.82rem; color: #D1D5DB;">Fast SMA Period:</label>
                <input type="number" id="st-param-fast" value="20" min="2" max="100" style="width:100%; padding:0.5rem; background:rgba(15,23,42,0.8); border:1px solid var(--border-color); border-radius:6px; color:#FFF; margin-top:0.25rem;">
            </div>
            <div class="form-group">
                <label style="font-size: 0.82rem; color: #D1D5DB;">Slow SMA Period:</label>
                <input type="number" id="st-param-slow" value="50" min="5" max="300" style="width:100%; padding:0.5rem; background:rgba(15,23,42,0.8); border:1px solid var(--border-color); border-radius:6px; color:#FFF; margin-top:0.25rem;">
            </div>
        `;
    } else if (template === "price_vs_sma") {
        container.innerHTML = `
            <div class="form-group" style="grid-column: span 2;">
                <label style="font-size: 0.82rem; color: #D1D5DB;">SMA Period (Cross Above):</label>
                <input type="number" id="st-param-period" value="50" min="5" max="300" style="width:100%; padding:0.5rem; background:rgba(15,23,42,0.8); border:1px solid var(--border-color); border-radius:6px; color:#FFF; margin-top:0.25rem;">
            </div>
        `;
    } else if (template === "volume_breakout") {
        container.innerHTML = `
            <div class="form-group" style="grid-column: span 2;">
                <label style="font-size: 0.82rem; color: #D1D5DB;">Volume Multiplier (vs 20D SMA):</label>
                <input type="number" id="st-param-multiplier" value="2.0" step="0.1" min="1.0" max="10.0" style="width:100%; padding:0.5rem; background:rgba(15,23,42,0.8); border:1px solid var(--border-color); border-radius:6px; color:#FFF; margin-top:0.25rem;">
            </div>
        `;
    }
}

async function handleRunStrategyTest() {
    const templateSelect = document.getElementById("st-template-select");
    const assetSelect = document.getElementById("st-asset-select");
    const exitInput = document.getElementById("st-exit-days-input");
    const runBtn = document.getElementById("btn-run-strategy-test");

    if (!templateSelect || !assetSelect) return;

    const template = templateSelect.value;
    const asset = assetSelect.value;
    const exitDays = parseInt(exitInput.value) || 10;

    const params = {};
    if (template === "rsi_threshold") {
        params.period = parseInt(document.getElementById("st-param-period").value) || 14;
        params.buy_threshold = parseFloat(document.getElementById("st-param-buy-threshold").value) || 30;
    } else if (template === "sma_crossover") {
        params.fast = parseInt(document.getElementById("st-param-fast").value) || 20;
        params.slow = parseInt(document.getElementById("st-param-slow").value) || 50;
    } else if (template === "price_vs_sma") {
        params.period = parseInt(document.getElementById("st-param-period").value) || 50;
    } else if (template === "volume_breakout") {
        params.multiplier = parseFloat(document.getElementById("st-param-multiplier").value) || 2.0;
    }

    if (runBtn) {
        runBtn.disabled = true;
        runBtn.innerText = "⏳ Running Strategy Simulation...";
    }

    try {
        const res = await fetch("/api/strategy-test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                asset: asset,
                template: template,
                params: params,
                exit_days: exitDays
            })
        });

        const data = await res.json();
        renderStrategyTestResults(data);
    } catch (err) {
        console.error("Failed to run strategy test:", err);
        alert("Error executing strategy test.");
    } finally {
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.innerText = "▶ RUN STRATEGY TEST";
        }
    }
}

function renderStrategyTestResults(data) {
    const banner = document.getElementById("st-verdict-banner");
    const tbody = document.getElementById("st-results-tbody");
    if (!banner || !tbody) return;

    if (data.beat_buy_and_hold) {
        banner.style.background = "rgba(16, 185, 129, 0.15)";
        banner.style.borderColor = "rgba(16, 185, 129, 0.4)";
        banner.style.color = "#34D399";
    } else {
        banner.style.background = "rgba(239, 68, 68, 0.15)";
        banner.style.borderColor = "rgba(239, 68, 68, 0.4)";
        banner.style.color = "#F87171";
    }
    banner.innerText = data.verdict;

    const sm = data.strategy_metrics;
    const bm = data.benchmark_metrics;

    const metricsMap = [
        { label: "CAGR (Annualized Return)", sVal: `${sm.cagr_pct.toFixed(2)}%`, bVal: `${bm.cagr_pct.toFixed(2)}%`, diff: sm.cagr_pct - bm.cagr_pct, unit: "%" },
        { label: "Sharpe Ratio (Risk-Adjusted)", sVal: sm.sharpe.toFixed(2), bVal: bm.sharpe.toFixed(2), diff: sm.sharpe - bm.sharpe, unit: "" },
        { label: "Sortino Ratio (Downside-Adjusted)", sVal: sm.sortino.toFixed(2), bVal: bm.sortino.toFixed(2), diff: sm.sortino - bm.sortino, unit: "" },
        { label: "Max Drawdown (Peak-to-Trough)", sVal: `${sm.max_dd_pct.toFixed(2)}%`, bVal: `${bm.max_dd_pct.toFixed(2)}%`, diff: bm.max_dd_pct - sm.max_dd_pct, unit: "%", lowerIsBetter: true },
        { label: "Total Trades Executed", sVal: sm.total_trades, bVal: `${bm.total_trades} (Buy & Hold)`, diff: null, unit: "" },
        { label: "Win Rate %", sVal: `${sm.win_rate_pct.toFixed(2)}%`, bVal: "N/A", diff: null, unit: "" },
    ];

    tbody.innerHTML = "";
    metricsMap.forEach(row => {
        const tr = document.createElement("tr");
        tr.style.borderBottom = "1px solid rgba(255, 255, 255, 0.05)";

        let diffText = "-";
        let diffClass = "";

        if (row.diff !== null) {
            const isGood = row.lowerIsBetter ? row.diff > 0 : row.diff > 0;
            diffClass = isGood ? "positive" : "negative";
            const sign = row.diff >= 0 ? "+" : "";
            diffText = `${sign}${row.diff.toFixed(2)}${row.unit}`;
        }

        tr.innerHTML = `
            <td style="padding: 0.6rem;"><strong>${row.label}</strong></td>
            <td style="padding: 0.6rem; color: #38BDF8; font-weight: 600;">${row.sVal}</td>
            <td style="padding: 0.6rem; color: #F59E0B;">${row.bVal}</td>
            <td style="padding: 0.6rem;" class="${diffClass}">${diffText}</td>
        `;
        tbody.appendChild(tr);
    });

    // Render Per-Asset Breakdown Table if available (ALL POOLED view)
    const breakdownContainer = document.getElementById("st-breakdown-container");
    const breakdownTbody = document.getElementById("st-breakdown-tbody");

    if (breakdownContainer && breakdownTbody) {
        if (data.per_asset_breakdown && data.per_asset_breakdown.length > 0) {
            breakdownContainer.style.display = "block";
            breakdownTbody.innerHTML = "";

            data.per_asset_breakdown.forEach(item => {
                const tr = document.createElement("tr");
                tr.style.borderBottom = "1px solid rgba(255, 255, 255, 0.05)";

                const diffClass = item.cagr_diff >= 0 ? "positive" : "negative";
                const sign = item.cagr_diff >= 0 ? "+" : "";
                const verdictBg = item.beat_bh ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)";
                const verdictColor = item.beat_bh ? "#34D399" : "#F87171";

                tr.innerHTML = `
                    <td style="padding: 0.45rem;"><strong>${item.asset_display_name}</strong></td>
                    <td style="padding: 0.45rem; color: #38BDF8; font-weight: 600;">${item.strategy_cagr_pct.toFixed(2)}%</td>
                    <td style="padding: 0.45rem; color: #F59E0B;">${item.bh_cagr_pct.toFixed(2)}%</td>
                    <td style="padding: 0.45rem;" class="${diffClass}">${sign}${item.cagr_diff.toFixed(2)}%</td>
                    <td style="padding: 0.45rem;">${item.total_trades}</td>
                    <td style="padding: 0.45rem;">${item.win_rate_pct.toFixed(2)}%</td>
                    <td style="padding: 0.45rem;">
                        <span style="padding: 0.15rem 0.4rem; border-radius: 4px; background: ${verdictBg}; color: ${verdictColor}; font-weight: 700; font-size: 0.75rem;">
                            ${item.verdict}
                        </span>
                    </td>
                `;
                breakdownTbody.appendChild(tr);
            });
        } else {
            breakdownContainer.style.display = "none";
            breakdownTbody.innerHTML = "";
        }
    }
}
