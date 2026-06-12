/**
 * Dashboard interactivity: market status, signal feed, WebSocket.
 */

const MARKET_STATUS = document.getElementById('market-status');
const STATUS_DOT = document.getElementById('status-dot');
const STATUS_TEXT = document.getElementById('status-text');
const MARKET_STATE = document.getElementById('market-state');
const MARKET_TIMES = document.getElementById('market-times');
const SIGNAL_FEED = document.getElementById('signal-feed');
const SIGNAL_COUNT = document.getElementById('signal-count');
const BULLISH_COUNT = document.getElementById('bullish-count');
const BEARISH_COUNT = document.getElementById('bearish-count');

let bullishTotal = 0;
let bearishTotal = 0;

// ─── Market Status ───────────────────────────────────────────────────────────

async function updateMarketStatus() {
    try {
        const resp = await fetch('/api/market-status');
        if (!resp.ok) return;
        const status = await resp.json();

        if (STATUS_DOT) {
            STATUS_DOT.className = 'status-dot ' + status.status.toLowerCase();
        }
        if (STATUS_TEXT) {
            STATUS_TEXT.textContent = status.status;
        }
        if (MARKET_STATE) {
            MARKET_STATE.textContent = status.status;
            MARKET_STATE.className = 'badge';
        }
        if (MARKET_TIMES) {
            let html = '';
            if (status.next_open) html += `<div>Next Open: ${status.next_open}</div>`;
            if (status.next_close) html += `<div>Next Close: ${status.next_close}</div>`;
            MARKET_TIMES.innerHTML = html;
        }
    } catch (err) {
        console.error('Market status error:', err);
    }
}

// ─── WebSocket ───────────────────────────────────────────────────────────────

let ws = null;
let wsReconnectTimer = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        // Send ping every 30s to keep alive
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) ws.send('ping');
        }, 30000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'signals') {
            handleNewSignals(data.signals);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting in 5s...');
        wsReconnectTimer = setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };
}

function handleNewSignals(signals) {
    if (!signals || signals.length === 0) return;

    for (const sig of signals) {
        const isBullish = sig.direction === 'BULLISH';
        if (isBullish) bullishTotal++;
        else bearishTotal++;

        if (SIGNAL_FEED) {
            const item = document.createElement('div');
            item.className = `signal-item ${isBullish ? 'bullish' : 'bearish'}`;
            item.innerHTML = `
                <div class="sig-header">
                    <span class="sig-ticker">${sig.ticker}</span>
                    <span class="sig-direction ${isBullish ? 'bullish' : 'bearish'}">
                        ${isBullish ? '▲ BULL' : '▼ BEAR'}
                    </span>
                </div>
                <div class="sig-strategy">${sig.strategy} | Conf: ${(sig.confidence * 100).toFixed(0)}%</div>
                <div class="sig-reasoning">${sig.reasoning || ''}</div>
            `;
            SIGNAL_FEED.prepend(item);

            // Keep max 50 items
            while (SIGNAL_FEED.children.length > 50) {
                SIGNAL_FEED.lastChild.remove();
            }
        }
    }

    updateCounts();
}

function updateCounts() {
    if (BULLISH_COUNT) BULLISH_COUNT.textContent = bullishTotal;
    if (BEARISH_COUNT) BEARISH_COUNT.textContent = bearishTotal;
    if (SIGNAL_FEED && SIGNAL_COUNT) {
        SIGNAL_COUNT.textContent = `${SIGNAL_FEED.children.length} signals`;
    }
}

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    updateMarketStatus();

    // WebSocket only on dashboard page
    if (SIGNAL_FEED) {
        connectWebSocket();
    }

    // Refresh market status every 60s
    setInterval(updateMarketStatus, 60000);
});
