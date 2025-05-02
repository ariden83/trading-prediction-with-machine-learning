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

// loadData pour charger les données 1d depuis un fichier JSON.
async function loadData(jsonFile) {
    try {
        const response = await fetch(jsonFile);
        logMessage('success', `Load JSON file (${jsonFile})`);
        const json = await response.json();
        // Supposons que les données sont dans json.dataPoints ou adapter selon la structure
        // Ici, on extrait les OHLCV pour chaque dataPoints
        const ohlcvData = [];
        if (json && Array.isArray(json.intervalsDataPoints)) {
            json.intervalsDataPoints.forEach(day => {
                if (Array.isArray(day.dataPoints)) {
                    day.dataPoints.forEach(point => {
                        ohlcvData.push({
                            timestamp: point.timestamp,
                            open: point.openPrice.ask,
                            high: point.highPrice.ask,
                            low: point.lowPrice.ask,
                            close: point.closePrice.ask,
                            volume: point.lastTradedVolume || 0
                        });
                    });
                }
            });
        }
        logMessage('success', `Données chargées (${ohlcvData.length} barres)`);
        return ohlcvData;
    } catch (error) {
        logMessage('error', `Erreur lors du chargement des données 1d: ${error.message}`);
    }
}

(async () => {
    sampleData['1d'] = await loadData('../../../brent/1d/all_v2.json');
    sampleData['4h'] = await loadData('../../../brent/4h/all_v2.json');
    sampleData['1h'] = await loadData('../../../brent/1h/all_v2.json');
    sampleData['5min'] = (await loadData('../../../brent/5min/2025-03-11-23.json'))
        .filter(d => d.close !== undefined && d.open !== undefined)
        .slice(-16)
        .map(d => ({ ...d, date: new Date(d.timestamp).toLocaleString() }));

    console.table( sampleData['5min']);
    console.table( sampleData['1h']);
    console.table( sampleData['4h']);
    console.table( sampleData['1d']);
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
    const data = {
        timeframe: timeframe,
        data: sampleData[timeframe]
    };

    logMessage('info', `Aperçu des premières valeurs pour ${timeframe}: ${JSON.stringify(data.data.slice(0, 3), null, 2)}`);
    socket.send(JSON.stringify(data));
    logMessage('info', `Données envoyées pour le timeframe ${timeframe}: ${data.data.length} barres`);
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

    logContainer.prepend(logEntry);

    // Limiter le nombre d'entrées de journal
    if (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

// Générer des données OHLCV aléatoires pour tester
function generateSampleData(count, timeframe) {
    const data = [];
    let basePrice = 100;
    let baseVolume = 1000;

    // Facteurs multiplicateurs en fonction du timeframe
    let priceFactor = 0.002; // 5min par défaut
    let volumeFactor = 1;

    switch (timeframe) {
        case '1h':
            priceFactor = 0.005;
            volumeFactor = 12;
            break;
        case '4h':
            priceFactor = 0.01;
            volumeFactor = 48;
            break;
        case '1d':
            priceFactor = 0.02;
            volumeFactor = 288;
            break;
    }

    // Déterminer le pas de temps en fonction du timeframe
    let timeStep = 5 * 60 * 1000; // 5 minutes en millisecondes par défaut

    switch (timeframe) {
        case '1h':
            timeStep = 60 * 60 * 1000; // 1 heure
            break;
        case '4h':
            timeStep = 4 * 60 * 60 * 1000; // 4 heures
            break;
        case '1d':
            timeStep = 24 * 60 * 60 * 1000; // 1 jour
            break;
    }

    // Date de départ (il y a count * timeStep millisecondes)
    let timestamp = Date.now() - (count * timeStep);

    for (let i = 0; i < count; i++) {
        // Variation aléatoire du prix
        const change = (Math.random() - 0.5) * 2 * priceFactor * basePrice;
        basePrice += change;

        // Calculer les valeurs OHLCV
        const open = basePrice;
        const high = open * (1 + Math.random() * priceFactor);
        const low = open * (1 - Math.random() * priceFactor);
        const close = (high + low) / 2;
        const volume = Math.floor(baseVolume * (0.8 + Math.random() * 0.4) * volumeFactor);

        data.push({
            timestamp: timestamp,
            open: open,
            high: high,
            low: low,
            close: close,
            volume: volume
        });

        // Avancer dans le temps
        timestamp += timeStep;
    }

    return data;
}