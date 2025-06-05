// Variables globales
let socket = null;
let messagesCount = 0;
let upPredictions = 0;
let downPredictions = 0;
let activeTimeframe = '5min';

// Données OHLCV simulées pour tester
const sampleData = {
    '5min': [],
    '1h': [],
    '4h': [],
    '1d': []
};

// Configuration globale
const config = {
    // URL de base pour les appels API
    apiBaseUrl: 'http://localhost:3001/api', // URL complète du serveur Express
    // Symbole du Brent sur Yahoo Finance
    brentSymbol: 'BZ=F',
    // Périodes disponibles pour l'affichage des données
    periods: {
        '5m': { interval: '5m', days: 1 },
        '5d': { interval: '30m', days: 5 },
        '1mo': { interval: '1d', days: 30 },
        '6mo': { interval: '1wk', days: 180 },
        '1y': { interval: '1mo', days: 365 }
    }
};


// loadData pour charger les données depuis un fichier JSON.
async function loadData(jsonFile) {
    try {
        console.log(`Tentative de chargement: ${jsonFile}`);
        const response = await fetch(jsonFile);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        logMessage('success', `Load JSON file (${jsonFile})`);
        const json = await response.json();
        console.log(`Contenu JSON pour ${jsonFile}:`, json);
        
        // Extraire les OHLCV - gérer trois structures différentes
        const ohlcvData = [];
        
        // Structure 1: {intervalsDataPoints: [...]} (anciens fichiers 5min)
        if (json && Array.isArray(json.intervalsDataPoints)) {
            console.log('Structure détectée: intervalsDataPoints');
            json.intervalsDataPoints.forEach((interval, intervalIndex) => {
                console.log(`Interval ${intervalIndex}: ${interval.dataPoints ? interval.dataPoints.length : 0} points`);
                
                if (Array.isArray(interval.dataPoints) && interval.dataPoints.length > 0) {
                    interval.dataPoints.forEach((point, pointIndex) => {
                        // Vérifier que le point a les données nécessaires
                        if (point.openPrice && point.highPrice && point.lowPrice && point.closePrice) {
                            ohlcvData.push({
                                timestamp: point.timestamp,
                                open: point.openPrice.ask || point.openPrice.bid || point.openPrice,
                                high: point.highPrice.ask || point.highPrice.bid || point.highPrice,
                                low: point.lowPrice.ask || point.lowPrice.bid || point.lowPrice,
                                close: point.closePrice.ask || point.closePrice.bid || point.closePrice,
                                volume: point.lastTradedVolume || 0
                            });
                        } else {
                            console.log(`Point ${pointIndex} dans interval ${intervalIndex} manque des données OHLC`);
                        }
                    });
                }
            });
        }
        // Structure 2: {Interval: '...', Candles: [{InstrumentId: 17, Candles: [...]}]} (anciens fichiers 1h, 4h, 1d)
        else if (json && Array.isArray(json.Candles)) {
            console.log('Structure détectée: Candles avec imbrication');
            console.log(`Nombre d'instruments: ${json.Candles.length}`);
            
            json.Candles.forEach((instrument, instrumentIndex) => {
                console.log(`Instrument ${instrumentIndex} (ID: ${instrument.InstrumentId}): ${instrument.Candles ? instrument.Candles.length : 0} candles`);
                
                if (Array.isArray(instrument.Candles)) {
                    instrument.Candles.forEach((candle, candleIndex) => {
                        // Vérifier que la bougie a les données nécessaires
                        if (candle.Open !== undefined && candle.High !== undefined && 
                            candle.Low !== undefined && candle.Close !== undefined) {
                            
                            // Convertir la date en timestamp
                            let timestamp = candle.FromDate;
                            if (typeof timestamp === 'string') {
                                timestamp = new Date(timestamp).getTime();
                            }
                            
                            ohlcvData.push({
                                timestamp: timestamp,
                                open: candle.Open,
                                high: candle.High,
                                low: candle.Low,
                                close: candle.Close,
                                volume: candle.Volume || 0
                            });
                        } else {
                            console.log(`Candle ${candleIndex} de l'instrument ${instrumentIndex} manque des données OHLC`);
                        }
                    });
                }
            });
        }
        // Structure 3: {timestamp: {timestamp, openPrice: {ask, bid}, ...}} (nouveaux fichiers live)
        else if (json && typeof json === 'object' && !Array.isArray(json)) {
            console.log('Structure détectée: Live data (timestamp as keys)');
            const timestamps = Object.keys(json);
            console.log(`Nombre de points de données live: ${timestamps.length}`);
            
            timestamps.forEach(timestampKey => {
                const point = json[timestampKey];
                
                // Vérifier que le point a les données nécessaires
                if (point.openPrice && point.highPrice && point.lowPrice && point.closePrice) {
                    ohlcvData.push({
                        timestamp: parseInt(timestampKey),
                        open: point.openPrice.ask || point.openPrice.bid,
                        high: point.highPrice.ask || point.highPrice.bid,
                        low: point.lowPrice.ask || point.lowPrice.bid,
                        close: point.closePrice.ask || point.closePrice.bid,
                        volume: point.lastTradedVolume || 0
                    });
                } else {
                    console.log(`Point ${timestampKey} manque des données OHLC`);
                }
            });
            
            // Trier par timestamp
            ohlcvData.sort((a, b) => a.timestamp - b.timestamp);
        }
        else {
            console.log('Structure JSON non reconnue:', Object.keys(json));
        }
        logMessage('success', `Données chargées (${ohlcvData.length} barres)`);
        return ohlcvData;
    } catch (error) {
        logMessage('error', `Erreur lors du chargement de ${jsonFile}: ${error.message}`);
        return null;
    }
}

(async () => {
    // Charger et filtrer les données pour tous les timeframes
    const rawData1d = await loadData('../live/1d_2025-6-2.json');
    sampleData['1d'] = rawData1d ? rawData1d
        .filter(d => d.close !== undefined && d.open !== undefined)
        .slice(-50) // Garder les 50 dernières données
        : [];

    const rawData4h = await loadData('../live/4h_2025-6-2.json');
    sampleData['4h'] = rawData4h ? rawData4h
        .filter(d => d.close !== undefined && d.open !== undefined)
        .slice(-50) // Garder les 50 dernières données
        : [];

    const rawData1h = await loadData('../live/1h_2025-6-2.json');
    sampleData['1h'] = rawData1h ? rawData1h
        .filter(d => d.close !== undefined && d.open !== undefined)
        .slice(-50) // Garder les 50 dernières données
        : [];

    const rawData5min = await loadData('../live/5m_2025-6-2.json');
    sampleData['5min'] = rawData5min ? rawData5min
        .filter(d => d.close !== undefined && d.open !== undefined)
        .slice(0, -22) // Retirer les 22 dernières données
        .slice(-50) // Garder les 50 dernières données
        .map(d => ({ ...d, date: new Date(d.timestamp).toLocaleString() }))
        : [];

    // Afficher les statistiques de chargement
    console.log('Données chargées:');
    console.log(`1d: ${sampleData['1d'].length} barres`);
    console.log(`4h: ${sampleData['4h'].length} barres`);
    console.log(`1h: ${sampleData['1h'].length} barres`);
    console.log(`5min: ${sampleData['5min'].length} barres`);

    if (sampleData['5min'].length > 0) {
        recreateChartWithData(sampleData['5min']);
    }
})();


// Éléments DOM
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const sendDataBtn = document.getElementById('sendDataBtn');
const serverUrlInput = document.getElementById('serverUrl');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const predictionText = document.getElementById('predictionText');
const confidenceMeter = document.getElementById('confidenceMeter');
const confidenceText = document.getElementById('confidenceText');
const lastUpdateTime = document.getElementById('lastUpdateTime');
const messagesCountElement = document.getElementById('messagesCount');
const upPredictionsElement = document.getElementById('upPredictions');
const downPredictionsElement = document.getElementById('downPredictions');
const predictionHistory = document.getElementById('predictionHistory');
const logContainer = document.getElementById('logContainer');
const predictionCard = document.getElementById('predictionCard');

// Sélecteurs de timeframe
const timeframeButtons = document.querySelectorAll('.timeframe-button');

// Initialisation
document.addEventListener('DOMContentLoaded', () => {

    // Gérer le bouton de connexion
    connectBtn.addEventListener('click', connectToServer);

    // Gérer le bouton de déconnexion
    disconnectBtn.addEventListener('click', disconnectFromServer);

    // Gérer le bouton d'envoi de données
    sendDataBtn.addEventListener('click', sendDataToServer);

    // Gérer les boutons de timeframe
    timeframeButtons.forEach(button => {
        button.addEventListener('click', () => {
            timeframeButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            activeTimeframe = button.dataset.timeframe;
            logMessage('info', `Timeframe actif changé pour ${activeTimeframe}`);
        });
    });

    // Log initial
    logMessage('info', 'Client WebSocket initialisé');
});

// Fonction pour se connecter au serveur WebSocket
function connectToServer() {
    const serverUrl = serverUrlInput.value;
    
    logMessage('info', `Clic sur le bouton "Se connecter" - URL: ${serverUrl}`);

    try {
        // Mettre à jour l'état
        statusIndicator.className = 'status-indicator connecting';
        statusText.textContent = 'Connexion en cours...';
        logMessage('info', `Tentative de connexion à ${serverUrl}`);

        // Créer la connexion WebSocket
        socket = new WebSocket(serverUrl);

        // Gérer les événements WebSocket
        socket.onopen = handleSocketOpen;
        socket.onmessage = handleSocketMessage;
        socket.onclose = handleSocketClose;
        socket.onerror = handleSocketError;

        // Désactiver/activer les boutons
        connectBtn.disabled = true;
        disconnectBtn.disabled = false;
    } catch (error) {
        logMessage('error', `Erreur de connexion: ${error.message}`);
        statusIndicator.className = 'status-indicator disconnected';
        statusText.textContent = 'Erreur de connexion';
    }
}

// Fonction pour se déconnecter du serveur
function disconnectFromServer() {
    logMessage('info', 'enter in disconnectFromServer');
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
        logMessage('info', 'Déconnexion manuelle du serveur');
    }
}

// Envoyer des données au serveur
function sendDataToServer() {
    logMessage('info', 'enter in sendDataToServer');
    if (socket && socket.readyState === WebSocket.OPEN) {
        logMessage('info', 'Envoi des données au serveur');
        // Envoyer toutes les timeframes à la fois
        sendTimeframeData('5min');
        sendTimeframeData('1h');
        sendTimeframeData('4h');
        sendTimeframeData('1d');

        logMessage('info', 'Données envoyées pour tous les timeframes');
        sendDataBtn.disabled = true; // Désactiver pour éviter les spam
        setTimeout(() => {
            sendDataBtn.disabled = false;
        }, 2000);
    } else {
        logMessage('error', 'Impossible d\'envoyer les données: non connecté');
    }
}

// Envoyer les données pour un timeframe spécifique
function sendTimeframeData(timeframe) {
    const timeframeData = sampleData[timeframe];
    
    // Debug: afficher les données avant envoi
    console.log(`DEBUG sendTimeframeData - ${timeframe}:`, {
        'sampleData[timeframe]': timeframeData,
        'longueur': timeframeData ? timeframeData.length : 'undefined',
        'type': typeof timeframeData
    });

    const data = {
        timeframe: timeframe,
        data: timeframeData || []
    };

    logMessage('info', `Envoi ${timeframe}: ${data.data.length} barres`);
    
    if (data.data.length > 0) {
        logMessage('info', `Aperçu ${timeframe}: ${JSON.stringify(data.data.slice(0, 2), null, 2)}`);
    } else {
        logMessage('warning', `Aucune donnée disponible pour ${timeframe}`);
    }
    
    socket.send(JSON.stringify(data));
}

// Gestion des événements WebSocket
function handleSocketOpen(event) {
    statusIndicator.className = 'status-indicator connected';
    statusText.textContent = 'Connecté';
    logMessage('success', 'Connexion établie avec le serveur');
    sendDataBtn.disabled = false;
}

function handleSocketMessage(event) {
    messagesCount++;
    messagesCountElement.textContent = messagesCount;

    try {
        const response = JSON.parse(event.data);
        logMessage('info', `Message reçu: ${event.data.substring(0, 100)}...`);

        if (response.status === 'success') {
            updatePrediction(response);
            addToPredictionHistory(response);
        } else {
            logMessage('error', `Erreur du serveur: ${response.message}`);
        }
    } catch (error) {
        logMessage('error', `Erreur de traitement du message: ${error.message}`);
    }
}

function handleSocketClose(event) {
    statusIndicator.className = 'status-indicator disconnected';
    statusText.textContent = 'Déconnecté';
    logMessage('info', `Connexion fermée. Code: ${event.code}, Raison: ${event.reason}`);

    // Réinitialiser les boutons
    connectBtn.disabled = false;
    disconnectBtn.disabled = true;
    sendDataBtn.disabled = true;

    // Réinitialiser le socket
    socket = null;
}

function handleSocketError(event) {
    statusIndicator.className = 'status-indicator disconnected';
    statusText.textContent = 'Erreur';
    logMessage('error', 'Erreur WebSocket');
}

// Mettre à jour l'affichage de la prédiction
function updatePrediction(response) {
    const prediction = response.prediction[0]; // La première valeur du tableau
    const timestamp = new Date(response.timestamp);

    // Mettre à jour le texte de prédiction
    if (prediction > 0.5) {
        predictionText.textContent = 'HAUSSE PRÉVUE';
        predictionCard.className = 'prediction-card prediction-up';
        upPredictions++;
    } else {
        predictionText.textContent = 'BAISSE PRÉVUE';
        predictionCard.className = 'prediction-card prediction-down';
        downPredictions++;
    }

    // Mettre à jour le compteur de prédictions
    upPredictionsElement.textContent = upPredictions;
    downPredictionsElement.textContent = downPredictions;

    // Mettre à jour l'indicateur de confiance
    const confidencePercent = prediction > 0.5 ? prediction * 100 : (1 - prediction) * 100;
    confidenceMeter.style.width = `${confidencePercent}%`;
    confidenceText.textContent = `Confiance: ${confidencePercent.toFixed(2)}%`;

    // Mettre à jour l'horodatage
    lastUpdateTime.textContent = timestamp.toLocaleString();
}

// Ajouter une prédiction à l'historique
function addToPredictionHistory(response) {
    const prediction = response.prediction[0];
    const timestamp = new Date(response.timestamp);
    const confidencePercent = prediction > 0.5 ? prediction * 100 : (1 - prediction) * 100;
    const direction = prediction > 0.5 ? 'HAUSSE' : 'BAISSE';

    const row = document.createElement('tr');
    row.innerHTML = `
                <td>${timestamp.toLocaleString()}</td>
                <td>${direction}</td>
                <td>${confidencePercent.toFixed(2)}%</td>
                <td>5min, 1h, 4h, 1d</td>
            `;

    predictionHistory.prepend(row);

    // Limiter le nombre d'éléments dans l'historique
    if (predictionHistory.children.length > 20) {
        predictionHistory.removeChild(predictionHistory.lastChild);
    }
}

// Ajouter un message au journal
function logMessage(type, message) {
    const now = new Date();
    const timeString = now.toLocaleTimeString();

    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    logEntry.innerHTML = `<span class="log-time">${timeString}</span> ${message}`;

    if (logContainer) {
        logContainer.prepend(logEntry);

        // Limiter le nombre d'entrées de journal
        if (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.lastChild);
        }
    } else {
        console.error('logContainer non trouvé!');
    }
}

function recreateChartWithData(data) {
    let chart; // Instance du graphique
    let currentPeriod = '5min'; // Période d'affichage par défaut

    // Récupération du canvas
    const chartContainer = document.querySelector('.chart-container');
    const existingCanvas = document.getElementById('price-chart');

    if (chart) {
        try {
            chart.destroy();
        } catch (e) {
            console.error('Erreur lors de la destruction du graphique:', e);
        }
        chart = null;
    }

    // Suppression du canvas existant
    if (existingCanvas && existingCanvas.parentNode) {
        existingCanvas.parentNode.removeChild(existingCanvas);
    }

    // Création d'un nouveau canvas
    const newCanvas = document.createElement('canvas');
    newCanvas.id = 'price-chart';
    chartContainer.appendChild(newCanvas);

    // Formatage des données pour le graphique en chandeliers
    const formattedData = data.map(item => ({
        t: new Date(item.timestamp),
        o: item.open,
        h: item.high,
        l: item.low,
        c: item.close
    }));

    // Tri des données par date (croissant)
    formattedData.sort((a, b) => a.t - b.t);

    // Création du nouveau graphique
    const ctx = newCanvas.getContext('2d');
    chart = new Chart(ctx, {
        type: 'candlestick',
        data: {
            datasets: [{
                label: 'Prix du Brent (USD)',
                data: formattedData,
                color: {
                    up: '#26a69a',
                    down: '#ef5350',
                    unchanged: '#888888',
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 500
            },
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return [
                                `Open: ${point.o}`,
                                `High: ${point.h}`,
                                `Low: ${point.l}`,
                                `Close: ${point.c}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    position: 'right',
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
        }
    });

    console.log('Graphique en chandeliers créé avec succès pour la période:', currentPeriod);
}

// Formater la date en fonction de la période
function formatDate(dateString, period) {
    // Création d'une date à partir de la chaîne
    const date = new Date(dateString);

    // Déterminer l'affichage en fonction de la période
    if (period === '5min') {
        // Vérifier l'intervalle actuel dans la configuration
        const interval = config.periods['5m'].interval;

        // Format HH:MM pour un intervalle de 5 minutes
        return date.toLocaleTimeString('fr-FR', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        });
    } else if (period === '5d') {
        // Format JJ/MM HHh pour un intervalle de 30 minutes
        return date.toLocaleDateString('fr-FR', {
                day: '2-digit',
                month: '2-digit'
            }) + ' ' +
            date.getHours().toString().padStart(2, '0') + 'h';
    } else if (period === '1mo') {
        // Format JJ/MM pour un intervalle d'un jour
        return date.toLocaleDateString('fr-FR', {
            day: '2-digit',
            month: '2-digit'
        });
    } else if (period === '6mo' || period === '1y') {
        // Format MM/AA pour les périodes plus longues
        return date.toLocaleDateString('fr-FR', {
            month: '2-digit',
            year: '2-digit'
        });
    } else {
        // Format par défaut
        return date.toLocaleDateString('fr-FR');
    }
}
