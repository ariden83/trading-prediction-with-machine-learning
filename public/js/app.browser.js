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
    sampleData['1d'] = (await loadData('../../../brent/1d/all_v2.json'));
    sampleData['4h'] = (await loadData('../../../brent/4h/all_v2.json'));
    sampleData['1h'] = (await loadData('../../../brent/1h/all_v2.json'));
    sampleData['5min'] = (await loadData('../../../brent/5min/2025-03-11-23.json'))
        .filter(d => d.close !== undefined && d.open !== undefined)
        .slice(0, -22) // Retirer les 20 dernières données
        .slice(-50) // Garder les 20 dernières données
        .map(d => ({ ...d, date: new Date(d.timestamp).toLocaleString() }));

    console.table( sampleData['5min']);
    //console.table( sampleData['1h']);
    //console.table( sampleData['4h']);
    //console.table( sampleData['1d']);
    recreateChartWithData(sampleData['5min'])
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

function recreateChartWithData(data) {
    let chart; // Instance du graphique
    let currentPeriod = '5min'; // Période d'affichage par défaut
    // console.log('Recréation complète du graphique avec période:', currentPeriod);
    // console.log('Données pour la recréation:', data);

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

    // Formatage des dates pour meilleure lisibilité
    const formattedData = [];
    for (let i = 0; i < data.length; i++) {
        formattedData.push({
            ...data[i],
            formattedDate: formatDate(data[i].timestamp, currentPeriod)
        });
    }


    console.table(formattedData);

    // Tri des données par date (croissant)
    formattedData.sort((a, b) => {
        return new Date(a.date) - new Date(b.date);
    });

    // Préparation des données pour le graphique
    const labels = formattedData.map(item => item.formattedDate);
    const prices = formattedData.map(item => item.close);

    console.log(`Préparation du graphique avec ${labels.length} labels et ${prices.length} prix.`);

    // Adapter le nombre de ticks en fonction de la période
    let ticksConfig = {};
    if (currentPeriod === '1m') {
        ticksConfig = {maxTicksLimit: 6};
    } else if (currentPeriod === '5min') {
        ticksConfig = { maxTicksLimit: 6 };
    } else if (currentPeriod === '5d') {
        ticksConfig = { maxTicksLimit: 10 };
    } else if (currentPeriod === '1mo') {
        ticksConfig = { maxTicksLimit: 15 };
    } else {
        ticksConfig = { maxTicksLimit: 12 };
    }

    // Création du nouveau graphique
    const ctx = newCanvas.getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Prix du Brent (USD)',
                data: prices,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 500 // Animation plus rapide
            },
            plugins: {
                legend: {
                    display: true // Afficher la légende pour distinguer les deux lignes
                },
            },
            scales: {
                y: {
                    beginAtZero: false
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });

    console.log('Graphique créé avec succès pour la période:', currentPeriod);
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
