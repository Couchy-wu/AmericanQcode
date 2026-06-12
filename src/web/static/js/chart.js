/**
 * Candlestick chart with Plotly.js and indicator overlays.
 * Used on Dashboard and Chart detail page.
 */

const CHART_TICKER = document.getElementById('chart-ticker');
const CHART_INTERVAL = document.getElementById('chart-interval');
const MAIN_CHART = document.getElementById('main-chart');
const DETAIL_CHART = document.getElementById('detail-chart');
const CHART_PRICE = document.getElementById('chart-price');

const chartContainer = MAIN_CHART || DETAIL_CHART;
const ticker = CHART_TICKER ? CHART_TICKER.value : (typeof TICKER !== 'undefined' ? TICKER : 'AAPL');
const interval = CHART_INTERVAL ? CHART_INTERVAL.value : '1d';

// Toggle checkboxes
const toggleMA = document.getElementById('toggle-ma');
const toggleBB = document.getElementById('toggle-bb');
const toggleMACD = document.getElementById('toggle-macd');
const toggleRSI = document.getElementById('toggle-rsi');

let currentData = null;

async function loadChart() {
    const tick = CHART_TICKER ? CHART_TICKER.value : ticker;
    const intvl = CHART_INTERVAL ? CHART_INTERVAL.value : interval;

    try {
        const resp = await fetch(`/api/chart/${tick}?interval=${intvl}&indicators=ma,macd,rsi,bollinger`);
        if (!resp.ok) {
            console.error('Failed to load chart data');
            return;
        }
        currentData = await resp.json();
        renderChart(currentData);

        if (CHART_PRICE) {
            const lastBar = currentData.bars[currentData.bars.length - 1];
            CHART_PRICE.textContent = `$${lastBar.close.toFixed(2)}`;
        }
    } catch (err) {
        console.error('Chart load error:', err);
    }
}

function renderChart(data) {
    const bars = data.bars;
    if (!bars || bars.length === 0) return;

    const dates = bars.map(b => b.timestamp);
    const opens = bars.map(b => b.open);
    const highs = bars.map(b => b.high);
    const lows = bars.map(b => b.low);
    const closes = bars.map(b => b.close);
    const volumes = bars.map(b => b.volume);

    const traces = [];

    // 1. Candlestick chart
    traces.push({
        x: dates, open: opens, high: highs, low: lows, close: closes,
        type: 'candlestick',
        name: ticker,
        increasing: { line: { color: '#00c853' }, fillcolor: '#00c853' },
        decreasing: { line: { color: '#ff1744' }, fillcolor: '#ff1744' },
        xaxis: 'x',
        yaxis: 'y',
        hoverlabel: { bgcolor: '#1a1a2e' },
    });

    // 2. Moving Averages (if available and toggled on)
    const showMA = toggleMA ? toggleMA.checked : true;
    const fastMA = bars[0] && 'MA_20' in bars[0] ? 'MA_20' : null;
    const slowMA = bars[0] && 'MA_50' in bars[0] ? 'MA_50' : null;

    if (showMA) {
        if (fastMA) {
            const vals = bars.map(b => b[fastMA] || null);
            traces.push({ x: dates, y: vals, type: 'scatter', mode: 'lines',
                name: 'MA20', line: { color: '#ffab00', width: 1 }, xaxis: 'x', yaxis: 'y' });
        }
        if (slowMA) {
            const vals = bars.map(b => b[slowMA] || null);
            traces.push({ x: dates, y: vals, type: 'scatter', mode: 'lines',
                name: 'MA50', line: { color: '#448aff', width: 1 }, xaxis: 'x', yaxis: 'y' });
        }
    }

    // 3. Bollinger Bands (if available and toggled on)
    const showBB = toggleBB ? toggleBB.checked : true;
    if (showBB && bars[0] && 'BB_Upper' in bars[0]) {
        const upperBB = bars.map(b => b.BB_Upper || null);
        const middleBB = bars.map(b => b.BB_Middle || null);
        const lowerBB = bars.map(b => b.BB_Lower || null);
        traces.push({ x: dates, y: upperBB, type: 'scatter', mode: 'lines',
            name: 'BB Upper', line: { color: 'rgba(255,255,255,0.2)', width: 1, dash: 'dash' }, xaxis: 'x', yaxis: 'y' });
        traces.push({ x: dates, y: middleBB, type: 'scatter', mode: 'lines',
            name: 'BB Middle', line: { color: 'rgba(255,255,255,0.3)', width: 1 }, xaxis: 'x', yaxis: 'y' });
        traces.push({ x: dates, y: lowerBB, type: 'scatter', mode: 'lines',
            name: 'BB Lower', line: { color: 'rgba(255,255,255,0.2)', width: 1, dash: 'dash' }, xaxis: 'x', yaxis: 'y',
            fill: 'tonexty', fillcolor: 'rgba(255,255,255,0.03)' });
    }

    // Subplots
    const hasMACD = bars[0] && 'MACD' in bars[0];
    const hasRSI = bars[0] && 'RSI' in bars[0];

    const layout = {
        template: 'plotly_dark',
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#16213e',
        font: { color: '#a0a0b0' },
        xaxis: {
            rangeslider: { visible: false },
            type: 'date',
            gridcolor: '#2a2a4a',
        },
        yaxis: {
            title: 'Price ($)',
            gridcolor: '#2a2a4a',
            side: 'right',
        },
        margin: { l: 60, r: 50, t: 30, b: 40 },
        showlegend: true,
        legend: {
            x: 0, y: 1, orientation: 'h',
            font: { size: 10, color: '#a0a0b0' },
        },
        height: 600,
    };

    if (hasMACD || hasRSI) {
        layout.grid = { rows: 3, columns: 1, roworder: 'top to bottom', pattern: 'independent' };
        layout.height = 800;

        // Main chart takes row 1, MACD row 2, RSI row 3
        layout.yaxis = { ...layout.yaxis, domain: [0.4, 1] };
        layout.xaxis = { ...layout.xaxis, domain: [0, 1], anchor: 'y' };

        let nextRow = 2;
        if (hasMACD) {
            const macdLine = bars.map(b => b.MACD || null);
            const signalLine = bars.map(b => b.MACD_Signal || null);
            const histogram = bars.map(b => b.MACD_Histogram || null);
            traces.push({ x: dates, y: macdLine, type: 'scatter', mode: 'lines',
                name: 'MACD', line: { color: '#448aff', width: 1 }, xaxis: `x${nextRow}`, yaxis: `y${nextRow}` });
            traces.push({ x: dates, y: signalLine, type: 'scatter', mode: 'lines',
                name: 'Signal', line: { color: '#ffab00', width: 1 }, xaxis: `x${nextRow}`, yaxis: `y${nextRow}` });
            traces.push({ x: dates, y: histogram, type: 'bar',
                name: 'Hist', marker: { color: histogram.map(v => v >= 0 ? '#00c853' : '#ff1744') },
                xaxis: `x${nextRow}`, yaxis: `y${nextRow}` });
            layout[`xaxis${nextRow}`] = { domain: [0, 1], anchor: `y${nextRow}`, gridcolor: '#2a2a4a' };
            layout[`yaxis${nextRow}`] = { domain: [0.2, 0.38], title: 'MACD', gridcolor: '#2a2a4a' };
            nextRow++;
        }

        if (hasRSI) {
            const rsiVals = bars.map(b => b.RSI || null);
            traces.push({ x: dates, y: rsiVals, type: 'scatter', mode: 'lines',
                name: 'RSI', line: { color: '#e040fb', width: 1.5 }, xaxis: `x${nextRow}`, yaxis: `y${nextRow}` });
            // 70/30 lines
            traces.push({ x: dates, y: Array(dates.length).fill(70), type: 'scatter', mode: 'lines',
                name: 'Overbought (70)', line: { color: 'rgba(255,23,68,0.3)', width: 1, dash: 'dash' },
                xaxis: `x${nextRow}`, yaxis: `y${nextRow}`, showlegend: false });
            traces.push({ x: dates, y: Array(dates.length).fill(30), type: 'scatter', mode: 'lines',
                name: 'Oversold (30)', line: { color: 'rgba(0,200,83,0.3)', width: 1, dash: 'dash' },
                xaxis: `x${nextRow}`, yaxis: `y${nextRow}`, showlegend: false });
            layout[`xaxis${nextRow}`] = { domain: [0, 1], anchor: `y${nextRow}`, gridcolor: '#2a2a4a' };
            layout[`yaxis${nextRow}`] = { domain: [0, 0.18], title: 'RSI', gridcolor: '#2a2a4a', range: [0, 100] };
        }
    }

    const container = MAIN_CHART || DETAIL_CHART;
    if (container) {
        Plotly.react(container, traces, layout, { responsive: true });
    }
}

// Event listeners
if (CHART_TICKER) CHART_TICKER.addEventListener('change', loadChart);
if (CHART_INTERVAL) CHART_INTERVAL.addEventListener('change', loadChart);
if (toggleMA) toggleMA.addEventListener('change', () => currentData && renderChart(currentData));
if (toggleBB) toggleBB.addEventListener('change', () => currentData && renderChart(currentData));
if (toggleMACD) toggleMACD.addEventListener('change', () => currentData && renderChart(currentData));
if (toggleRSI) toggleRSI.addEventListener('change', () => currentData && renderChart(currentData));

// Initial load
if (chartContainer) {
    loadChart();
    // Auto refresh every 5 min for intraday
    setInterval(loadChart, 5 * 60 * 1000);
}
