/* AlphaForge Dashboard Application Logic & REST Integration */

let currentChartType = "area"; // "area" | "candlestick"
let currentChartTF = "3M";     // "1M" | "3M" | "6M" | "1Y" | "ALL"
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
            if (currentRawMarketData) renderInteractiveChart(currentRawMarketData);
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
            if (currentRawMarketData) renderInteractiveChart(currentRawMarketData);
        });
    });

    const smaBtn = document.getElementById("btn-toggle-sma");
    if (smaBtn) {
        smaBtn.addEventListener("click", () => {
            showSMAOverlay = !showSMAOverlay;
            smaBtn.style.background = showSMAOverlay ? "#6366F1" : "transparent";
            smaBtn.style.color = showSMAOverlay ? "#FFF" : "#9CA3AF";
            if (currentRawMarketData) renderInteractiveChart(currentRawMarketData);
        });
    }

    const volBtn = document.getElementById("btn-toggle-vol");
    if (volBtn) {
        volBtn.addEventListener("click", () => {
            showVolOverlay = !showVolOverlay;
            volBtn.style.background = showVolOverlay ? "#6366F1" : "transparent";
            volBtn.style.color = showVolOverlay ? "#FFF" : "#9CA3AF";
            if (currentRawMarketData) renderInteractiveChart(currentRawMarketData);
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
            btn.innerText = `${asset.display_name.split(' ')[0]} (₹${asset.last_price.toFixed(1)})`;
            btn.addEventListener("click", async () => {
                document.querySelectorAll(".asset-tab-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                currentAsset = asset.symbol;
                await refreshCurrentAssetView(currentAsset);
            });
            container.appendChild(btn);
        });
    } catch (err) {
        console.error("Failed to load assets:", err);
    }
}

function refreshCurrentAssetView(symbol) {
    loadChartData(symbol);
    loadSignalData(symbol);
    loadLivePrice(symbol);
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
        const res = await fetch(`/api/market-data?symbol=${symbol}`);
        const data = await res.json();
        renderInteractiveChart(data);
    } catch (err) {
        console.error("Failed to load chart data:", err);
    }
}

function renderInteractiveChart(data) {
    currentRawMarketData = data;
    const totalBars = data.dates.length;
    let barCount = totalBars;

    if (currentChartTF === "1M") barCount = 22;
    else if (currentChartTF === "3M") barCount = 66;
    else if (currentChartTF === "6M") barCount = 132;
    else if (currentChartTF === "1Y") barCount = 252;
    else if (currentChartTF === "ALL") barCount = totalBars;

    const startIdx = Math.max(0, totalBars - barCount);

    let labels = data.dates.slice(startIdx);
    let openPrices = (data.open && data.open.length === totalBars ? data.open : data.close).slice(startIdx);
    let highPrices = (data.high && data.high.length === totalBars ? data.high : data.close).slice(startIdx);
    let lowPrices = (data.low && data.low.length === totalBars ? data.low : data.close).slice(startIdx);
    let closePrices = data.close.slice(startIdx);
    let volumes = (data.volume && data.volume.length === totalBars ? data.volume : labels.map(() => 1000)).slice(startIdx);
    let sma20Data = data.sma20 ? data.sma20.slice(startIdx) : [];
    let sma50Data = data.sma50 ? data.sma50.slice(startIdx) : [];

    let isLiveActive = false;
    if (data.live_price_info && data.live_price_info.current_price) {
        const liveP = data.live_price_info.current_price;
        const prevC = data.live_price_info.previous_close || closePrices[closePrices.length - 1];
        labels.push("LIVE (Forming)");
        openPrices.push(prevC);
        highPrices.push(Math.max(liveP, prevC));
        lowPrices.push(Math.min(liveP, prevC));
        closePrices.push(liveP);
        volumes.push(volumes.length > 0 ? volumes[volumes.length - 1] : 1000);
        if (sma20Data.length > 0) sma20Data.push(sma20Data[sma20Data.length - 1]);
        if (sma50Data.length > 0) sma50Data.push(sma50Data[sma50Data.length - 1]);
        isLiveActive = true;
    }

    const firstClose = closePrices[0] || 1.0;
    const latestClose = closePrices[closePrices.length - 1] || 1.0;
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

    const lastIdx = closePrices.length - 1;
    updateOHLCHeader(labels[lastIdx], openPrices[lastIdx], highPrices[lastIdx], lowPrices[lastIdx], closePrices[lastIdx], volumes[lastIdx]);

    const canvasEl = document.getElementById("market-chart-canvas");
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");
    if (marketChart) marketChart.destroy();

    const isBullish = tfReturnPct >= 0;
    const primaryColor = isBullish ? "#10B981" : "#EF4444";

    const gradient = ctx.createLinearGradient(0, 0, 0, 320);
    if (isBullish) {
        gradient.addColorStop(0, "rgba(16, 185, 129, 0.35)");
        gradient.addColorStop(0.7, "rgba(16, 185, 129, 0.05)");
        gradient.addColorStop(1, "rgba(16, 185, 129, 0.0)");
    } else {
        gradient.addColorStop(0, "rgba(239, 68, 68, 0.35)");
        gradient.addColorStop(0.7, "rgba(239, 68, 68, 0.05)");
        gradient.addColorStop(1, "rgba(239, 68, 68, 0.0)");
    }

    const volBarColors = closePrices.map((c, i) => {
        const o = openPrices[i] || c;
        return c >= o ? "rgba(16, 185, 129, 0.45)" : "rgba(239, 68, 68, 0.45)";
    });

    const datasets = [];

    if (showVolOverlay) {
        datasets.push({
            type: 'bar',
            label: 'Volume',
            data: volumes,
            backgroundColor: volBarColors,
            borderColor: 'transparent',
            borderWidth: 0,
            yAxisID: 'yVolume',
            order: 3,
        });
    }

    if (currentChartType === "area") {
        const pointRadii = labels.map((l, i) => (i === labels.length - 1 && isLiveActive ? 6 : 0));
        const pointColors = labels.map((l, i) => (i === labels.length - 1 && isLiveActive ? '#F59E0B' : primaryColor));

        datasets.push({
            type: 'line',
            label: 'Close Price (₹)',
            data: closePrices,
            borderColor: primaryColor,
            borderWidth: 2.2,
            backgroundColor: gradient,
            fill: true,
            tension: 0.15,
            pointRadius: pointRadii,
            pointBackgroundColor: pointColors,
            pointBorderColor: '#FFFFFF',
            pointBorderWidth: 1.5,
            yAxisID: 'yPrice',
            order: 1,
        });
    } else {
        const candleColors = closePrices.map((c, i) => (c >= openPrices[i] ? '#10B981' : '#EF4444'));
        datasets.push({
            type: 'bar',
            label: 'Candle Body',
            data: closePrices.map((c, i) => [Math.min(openPrices[i], c), Math.max(openPrices[i], c)]),
            backgroundColor: candleColors,
            borderColor: candleColors,
            borderWidth: 1,
            barPercentage: 0.6,
            yAxisID: 'yPrice',
            order: 1,
        });
    }

    if (showSMAOverlay && sma20Data.length > 0) {
        datasets.push({
            type: 'line',
            label: 'SMA 20',
            data: sma20Data,
            borderColor: '#6366F1',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
            yAxisID: 'yPrice',
            order: 2,
        });
    }

    if (showSMAOverlay && sma50Data.length > 0) {
        datasets.push({
            type: 'line',
            label: 'SMA 50',
            data: sma50Data,
            borderColor: '#F59E0B',
            borderWidth: 1.5,
            borderDash: [2, 2],
            pointRadius: 0,
            fill: false,
            yAxisID: 'yPrice',
            order: 2,
        });
    }

    const maxVol = Math.max(...volumes, 1.0);

    marketChart = new Chart(ctx, {
        data: {
            labels: labels,
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            onHover: (event, elements) => {
                if (elements && elements.length > 0) {
                    const idx = elements[0].index;
                    updateOHLCHeader(labels[idx], openPrices[idx], highPrices[idx], lowPrices[idx], closePrices[idx], volumes[idx]);
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#F3F4F6',
                    bodyColor: '#D1D5DB',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: (ctxItem) => {
                            const val = ctxItem.raw;
                            if (ctxItem.dataset.label === 'Volume') {
                                return `Volume: ${val >= 1e6 ? (val/1e6).toFixed(2)+'M' : (val/1e3).toFixed(1)+'K'}`;
                            }
                            if (Array.isArray(val)) {
                                return `Candle Body: ₹${val[0].toFixed(2)} - ₹${val[1].toFixed(2)}`;
                            }
                            return `${ctxItem.dataset.label}: ₹${typeof val === 'number' ? val.toFixed(2) : val}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9CA3AF', maxTicksLimit: 12, font: { size: 10 } }
                },
                yPrice: {
                    type: 'linear',
                    position: 'right',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#9CA3AF',
                        font: { size: 10 },
                        callback: (v) => `₹${v.toLocaleString('en-IN')}`
                    }
                },
                yVolume: {
                    type: 'linear',
                    position: 'left',
                    display: false,
                    max: maxVol * 4,
                }
            }
        }
    });
}

function updateOHLCHeader(dateStr, o, h, l, c, v) {
    const dEl = document.getElementById("ohlc-date");
    const oEl = document.getElementById("ohlc-open");
    const hEl = document.getElementById("ohlc-high");
    const lEl = document.getElementById("ohlc-low");
    const cEl = document.getElementById("ohlc-close");
    const vEl = document.getElementById("ohlc-vol");

    if (dEl) dEl.innerText = dateStr || "-";
    if (oEl) oEl.innerText = o ? `₹${o.toFixed(2)}` : "-";
    if (hEl) hEl.innerText = h ? `₹${h.toFixed(2)}` : "-";
    if (lEl) lEl.innerText = l ? `₹${l.toFixed(2)}` : "-";
    if (cEl) cEl.innerText = c ? `₹${c.toFixed(2)}` : "-";
    if (vEl) {
        vEl.innerText = v ? (v >= 1e6 ? `${(v/1e6).toFixed(2)}M` : `${(v/1e3).toFixed(1)}K`) : "-";
    }
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
