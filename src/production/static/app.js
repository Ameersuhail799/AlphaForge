/* AlphaForge Dashboard Application Logic & REST Integration */

let currentAsset = "tcs_ns";
let marketChart = null;

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    setupTabNavigation();
    setupEventListeners();
    await loadSupportedAssets();
    await refreshCurrentAssetView(currentAsset);
    await refreshPortfolioView();
    await loadBacktestResearchSummaries();
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

async function refreshCurrentAssetView(symbol) {
    await loadSignalData(symbol);
    await loadChartData(symbol);
}

async function loadSignalData(symbol) {
    try {
        const res = await fetch(`/api/signal?symbol=${symbol}`);
        const data = await res.json();

        document.getElementById("signal-asset-name").innerText = data.display_name;
        document.getElementById("signal-asset-symbol").innerText = data.symbol.toUpperCase();
        document.getElementById("signal-last-price").innerText = `₹${data.last_price.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

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

        const ctx = document.getElementById("market-chart-canvas").getContext("2d");
        if (marketChart) marketChart.destroy();

        marketChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: 'Close Price (₹)',
                        data: data.close,
                        borderColor: '#06B6D4',
                        borderWidth: 2,
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: 'SMA 20',
                        data: data.sma20,
                        borderColor: '#6366F1',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 0
                    },
                    {
                        label: 'SMA 50',
                        data: data.sma50,
                        borderColor: '#F59E0B',
                        borderWidth: 1.5,
                        borderDash: [2, 2],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9CA3AF', maxTicksLimit: 10 }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9CA3AF' }
                    }
                }
            }
        });
    } catch (err) {
        console.error("Failed to load chart data:", err);
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

async function refreshPortfolioView() {
    try {
        const res = await fetch("/api/portfolio");
        const data = await res.json();

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
        const tbody = document.getElementById("backtest-summary-tbody");
        tbody.innerHTML = "";

        data.mission27_cross_asset.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${row.asset.toUpperCase()}</strong></td>
                <td>${row.display_name}</td>
                <td class="positive">+${row.cum_return_pct.toFixed(2)}%</td>
                <td>${row.sharpe.toFixed(2)}</td>
                <td>+${row.expectancy_pct.toFixed(2)}% / trade</td>
                <td><span style="color: #10B981; font-weight: 700;">${row.positive_folds} Folds Positive</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error("Failed to load backtest summaries:", err);
    }
}
