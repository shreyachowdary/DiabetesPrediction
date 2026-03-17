/**
 * Diabetes Prediction - Frontend logic
 * Handles form submission, API calls, and result display
 */

const API_BASE = '';  // Same origin when deployed; use '' for relative URLs

async function getModelInfo() {
    try {
        const res = await fetch(`${API_BASE}/model-info`);
        if (!res.ok) throw new Error('Failed to fetch model info');
        const data = await res.json();
        renderModelInfo(data);
    } catch (err) {
        document.getElementById('model-info-content').innerHTML =
            `<p class="error">Could not load model info: ${err.message}</p>`;
    }
}

function renderModelInfo(data) {
    const container = document.getElementById('model-info-content');
    const features = data.selected_features || [];
    const metrics = data.metrics || {};
    const bestModel = data.best_model || 'N/A';

    let html = `
        <p><strong>Best Model:</strong> ${bestModel}</p>
        <p class="feature-list"><strong>Selected Features (GA):</strong> ${features.join(', ')}</p>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td>${(metrics.accuracy * 100 || 0).toFixed(2)}%</td></tr>
            <tr><td>Precision</td><td>${(metrics.precision * 100 || 0).toFixed(2)}%</td></tr>
            <tr><td>Recall</td><td>${(metrics.recall * 100 || 0).toFixed(2)}%</td></tr>
            <tr><td>F1-Score</td><td>${(metrics.f1_score * 100 || 0).toFixed(2)}%</td></tr>
            <tr><td>ROC-AUC</td><td>${(metrics.roc_auc || 0).toFixed(3)}</td></tr>
        </table>
    `;
    container.innerHTML = html;
}

async function predict(formData) {
    const payload = {};
    const fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ];
    fields.forEach(f => {
        const val = formData.get(f);
        payload[f] = f.includes('Diabetes') || ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'].includes(f)
            ? parseFloat(val) : parseInt(val, 10);
    });

    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed: ${res.status}`);
    }
    return res.json();
}

function showResult(data) {
    const section = document.getElementById('result-section');
    const content = document.getElementById('result-content');
    const isDiabetic = data.prediction === 'diabetic';
    const cls = isDiabetic ? 'result-diabetic' : 'result-non-diabetic';
    const label = isDiabetic ? 'Diabetic' : 'Non-Diabetic';
    const conf = (data.confidence * 100).toFixed(1);

    content.innerHTML = `
        <p class="${cls}">Prediction: ${label}</p>
        <p class="confidence">Confidence: ${conf}% (Probability: ${(data.probability * 100).toFixed(1)}%)</p>
    `;
    section.style.display = 'block';
}

document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('submit-btn');
    const form = e.target;
    const formData = new FormData(form);

    btn.disabled = true;
    btn.textContent = 'Predicting...';

    try {
        const result = await predict(formData);
        showResult(result);
    } catch (err) {
        document.getElementById('result-content').innerHTML =
            `<p class="error">${err.message}</p>`;
        document.getElementById('result-section').style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict';
    }
});

// Load model info on page load
getModelInfo();
