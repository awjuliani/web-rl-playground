import {
    canvas, ctx, loadImages, updateGridSize, resetAgent, takeAction,
    drawGrid, drawAgent, drawCellStates, drawValues,
    agentPos, // Import agentPos directly
    setAgentPos, // Import the setter function
    setStartPos, // Import start position setter
    setTerminateOnGem, cycleCellState,
    drawRewardText, // Import the new drawing function
    setStepPenalty, // Added setStepPenalty
    initializeGridRewards, // Import the initialization function
    drawPolicyArrows, // Import policy drawing function
    interpolateProbColor, // Import the color interpolation function
    drawSRVector, // NEW: Import SR vector drawing function
    setGemRewardMagnitude, // NEW: Import gem reward setter
    setBadStateRewardMagnitude, // NEW: Import bad state reward setter
    rewardMagnitudeGem as initialGemReward, // NEW: Import initial gem reward
    rewardMagnitudeBad as initialBadReward, // NEW: Import initial bad reward
    drawSRWVector, // NEW: Import SR W vector drawing function
} from './environment.js';

import {
    qTable, vTable, hTable, mTable, wTable,
    // Use getter functions instead of direct imports
    getLearningRate, getDiscountFactor, getExplorationRate, getSoftmaxBeta,
    getExplorationStrategy, getSelectedAlgorithm,
    // Keep the update functions
    initializeTables, learningStep,
    updateLearningRate, updateDiscountFactor, updateExplorationRate, updateSoftmaxBeta, updateExplorationStrategy,
    updateSelectedAlgorithm, applyMonteCarloUpdates,
    getActionProbabilities, getBestActions, calculateQValueSR,
    // Keep specific learning rates and their update functions
    getActorLearningRate, getCriticLearningRate,
    getSRMWeightLearningRate, getSRWWeightLearningRate,
    updateActorLearningRate, updateCriticLearningRate,
    updateSRMLearningRate, updateSRWLearningRate,
} from './algorithms.js';

// --- State Variables ---
let gridSize = 5; // Updated default to match HTML input
let cellSize = canvas.width / gridSize;
let terminateOnGem = true; // Default value
let simulationSpeed = 100; // Default speed (ms per step)
let animationDuration = 80; // Duration of the move animation (ms)
let maxStepsPerEpisode = 100;
let currentEpisodeSteps = 0;
let currentTheme = 'light'; // NEW: Track current theme

let learningInterval = null;
let isLearning = false;
let isAnimating = false;
let visualAgentPos = { x: 0, y: 0 }; // For rendering during animation (initialized in reset/initializeApp)
let animationFrameId = null;
let hoveredCell = null;
let cellDisplayMode = 'values-color';

// State for reward text animation
let rewardAnimation = { text: '', pos: null, alpha: 0, offsetY: 0, startTime: 0, duration: 600 }; // Duration in ms
let rewardAnimationFrameId = null;

let episodeCounter = 0;
let totalRewardForEpisode = 0;
let episodicRewards = [];
let smoothedEpisodicRewards = [];
let episodeNumbers = [];
const MOVING_AVERAGE_WINDOW = 20; // How many episodes to average over
const MAX_CHART_POINTS = 500; // Max points to display on the chart

// Reward chart state
let rewardChartInstance = null;
let rewardChartCtx = null;

// --- NEW: Explanation data state ---
let explanations = null; // Will be loaded from JSON

// --- DOM Elements ---
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const resetAgentButton = document.getElementById('resetAgentButton');
const resetEnvironmentButton = document.getElementById('resetEnvironmentButton');
const lrSlider = document.getElementById('lrSlider');
const lrValueSpan = document.getElementById('lrValue');
const lrControl = lrSlider.parentElement.parentElement; // Get the whole field container for the main LR slider
const discountSlider = document.getElementById('discountSlider');
const discountValueSpan = document.getElementById('discountValue');
const epsilonSlider = document.getElementById('epsilonSlider');
const epsilonValueSpan = document.getElementById('epsilonValue');
const explorationStrategySelect = document.getElementById('explorationStrategySelect');
const softmaxBetaControl = document.getElementById('softmaxBetaControl');
const softmaxBetaSlider = document.getElementById('softmaxBetaSlider');
const softmaxBetaValueSpan = document.getElementById('softmaxBetaValue');
const gridSizeSlider = document.getElementById('gridSizeSlider');
const gridSizeValueSpan = document.getElementById('gridSizeValue');
const algorithmSelect = document.getElementById('algorithmSelect');
const terminateOnRewardCheckbox = document.getElementById('terminateOnRewardCheckbox');
const speedSlider = document.getElementById('speedSlider');
const speedValueSpan = document.getElementById('speedValue');
const qValueDisplayDiv = document.getElementById('qValueDisplay');
const qValueDisplayHeader = document.querySelector('#qValueDisplay .collapsible-header');
const qGridDiv = document.querySelector('#qValueDisplay .q-grid');
const qUpSpan = document.getElementById('qUp');
const qDownSpan = document.getElementById('qDown');
const qLeftSpan = document.getElementById('qLeft');
const qRightSpan = document.getElementById('qRight');
const stepPenaltySlider = document.getElementById('stepPenaltySlider');
const stepPenaltyValueSpan = document.getElementById('stepPenaltyValue');
const cellDisplayModeSelect = document.getElementById('cellDisplayModeSelect');
const pUpSpan = document.getElementById('pUp');
const pDownSpan = document.getElementById('pDown');
const pLeftSpan = document.getElementById('pLeft');
const pRightSpan = document.getElementById('pRight');
const explanationTitle = document.getElementById('explanationTitle');
const algorithmExplanationDiv = document.getElementById('algorithmExplanation');
const explorationExplanationDiv = document.getElementById('explorationExplanation');
const maxStepsSlider = document.getElementById('maxStepsSlider');
const maxStepsValueSpan = document.getElementById('maxStepsValue');
const rewardChartCanvas = document.getElementById('rewardChartCanvas');
const srVectorAgentDisplayOption = document.getElementById('srVectorAgentDisplayOption');
const srVectorHoverDisplayOption = document.getElementById('srVectorHoverDisplayOption');
const themeToggleCheckbox = document.getElementById('theme-checkbox');
const gemRewardSlider = document.getElementById('gemRewardSlider');
const gemRewardValueSpan = document.getElementById('gemRewardValue');
const badStateRewardSlider = document.getElementById('badStateRewardSlider');
const badStateRewardValueSpan = document.getElementById('badStateRewardValue');

// Actor-Critic LR controls DOM Elements
const actorCriticLRControl = document.getElementById('actorCriticLRControl');
const actorLrSlider = document.getElementById('actorLrSlider');
const actorLrValueSpan = document.getElementById('actorLrValue');
const criticLRControl = document.getElementById('criticLRControl');
const criticLrSlider = document.getElementById('criticLrSlider');
const criticLrValueSpan = document.getElementById('criticLrValue');

// SR LR controls DOM Elements
const srMLRControl = document.getElementById('srMLRControl');
const srMLrSlider = document.getElementById('srMLrSlider');
const srMLrValueSpan = document.getElementById('srMLrValue');
const srWLRControl = document.getElementById('srWLRControl');
const srWLrSlider = document.getElementById('srWLrSlider');
const srWLrValueSpan = document.getElementById('srWLrValue');
const srWVectorDisplayOption = document.getElementById('srWVectorDisplayOption');

// --- NEW: Collapsible Settings Handler ---
function initializeCollapsibles() {
    const headers = document.querySelectorAll('.collapsible-header[data-toggle="collapse"]');
    headers.forEach(header => {
        const targetId = header.getAttribute('data-target');
        const targetContent = document.querySelector(targetId);

        if (targetContent) {
            // Initialize state (start expanded)
            // If you want them to start collapsed, add 'collapsed' class here initially
            // targetContent.classList.add('collapsed');
            // header.classList.add('collapsed');

            header.addEventListener('click', () => {
                const isCollapsed = targetContent.classList.contains('collapsed');

                if (isCollapsed) {
                    targetContent.classList.remove('collapsed');
                    header.classList.remove('collapsed');
                } else {
                    targetContent.classList.add('collapsed');
                    header.classList.add('collapsed');
                }
            });
        } else {
            console.warn(`Collapsible target not found: ${targetId}`);
        }
    });
}
// --- End NEW ---

// --- Educational Explanations ---
// REMOVED: explanations object definition moved to explanations.json

// Helper function for MathJax rendering
function renderMath(element) {
    if (typeof renderMathInElement === 'function') {
        try {
            renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError : false
            });
        } catch (error) {
            console.error("KaTeX rendering failed:", error);
        }
    } else {
        console.warn("renderMathInElement function not found. KaTeX might not be loaded yet.");
    }
}

// Update explanation text function
function updateExplanationText() {
    // Check if explanations are loaded
    if (!explanations) {
        console.warn("Explanations not loaded yet.");
        // explanationTitle.textContent = 'Loading Explanations...'; // Removed: Title is now static HTML
        algorithmExplanationDiv.innerHTML = '<p>Loading explanation content...</p>'; // Keep fallback content
        explorationExplanationDiv.innerHTML = '';
        return;
    }

    const algoKey = getSelectedAlgorithm();
    const strategyKey = getExplorationStrategy();

    const algoInfo = explanations.algorithms[algoKey];
    const strategyInfo = explanations.strategies[strategyKey];

    if (algoInfo) {
        // explanationTitle.textContent = `${algoInfo.title} Explanation`; // Removed: Title is now static HTML
        // Prepend the algorithm title before the explanation text
        algorithmExplanationDiv.innerHTML = `<h4>${algoInfo.title}</h4>${algoInfo.text}`;
    } else {
        // explanationTitle.textContent = 'Algorithm Explanation'; // Removed: Title is now static HTML
        algorithmExplanationDiv.innerHTML = '<h4>Algorithm Details</h4><p>Select an algorithm to see details.</p>';
    }

    if (algoKey === 'actor-critic') {
        explorationExplanationDiv.style.display = ''; // Ensure div is visible
        explorationExplanationDiv.innerHTML = `
            <h4>Policy-Based Exploration</h4>
            <p>Actor-Critic explores implicitly through its stochastic policy \\(\\pi(a|s)\\), which is typically derived from preferences \\(h(s,a)\\) using a softmax function:</p>
            $$ P(a|s) = \\frac{\\exp(\\beta h(s,a))}{\\sum_{a'} \\exp(\\beta h(s,a'))} $$
            <p>Actions with higher preferences are more likely, but all actions have a non-zero probability of being selected (unless \\(\\beta\\) is extremely high). Adjusting the Softmax Temperature \\(\\beta\\) controls the greediness of the policy and thus the degree of exploration.</p>
        `;
    } else {
        explorationExplanationDiv.style.display = ''; // Ensure div is visible
        if (strategyInfo) {
            explorationExplanationDiv.innerHTML = `<h4>${strategyInfo.title}</h4>` + strategyInfo.text;
        } else {
            explorationExplanationDiv.innerHTML = '';
        }
    }

    renderMath(algorithmExplanationDiv);
    renderMath(explorationExplanationDiv);
}
// --- End Educational Explanations ---

// --- Drawing Function ---
function drawEverything() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const algorithm = getSelectedAlgorithm();

    // 1. Draw Cell Background/Policy/SR Vector OR Nothing
    if (cellDisplayMode === 'values-color') {
        drawValues(ctx, gridSize, cellSize, qTable, vTable, mTable, wTable, algorithm, false);
    } else if (cellDisplayMode === 'values-text') {
        drawValues(ctx, gridSize, cellSize, qTable, vTable, mTable, wTable, algorithm, true);
    } else if (cellDisplayMode === 'policy') {
        drawPolicyArrows(ctx, gridSize, cellSize, qTable, hTable, mTable, wTable, algorithm, takeAction, agentPos);
    } else if (cellDisplayMode === 'sr-vector') { // Agent-based SR
        const agentStateKey = `${agentPos.x},${agentPos.y}`;
        drawSRVector(ctx, gridSize, cellSize, agentStateKey, mTable, true);
    } else if (cellDisplayMode === 'sr-vector-hover') { // Hover-based SR (NEW)
        if (hoveredCell) {
            const hoveredStateKey = `${hoveredCell.x},${hoveredCell.y}`;
            drawSRVector(ctx, gridSize, cellSize, hoveredStateKey, mTable, true);
        } else {
            // Optionally draw a placeholder if nothing is hovered
        }
    } else if (cellDisplayMode === 'sr-w-vector') { // NEW: W vector display
        drawSRWVector(ctx, gridSize, cellSize, wTable, true);
    }
    // else (cellDisplayMode === 'none') -> grid lines below will show through

    // 2. Draw Grid Lines & Start Icon & Hover Highlight (drawGrid handles these)
    drawGrid(gridSize, cellSize, hoveredCell);

    // 3. Draw Items (Gems/Bad States) - On top of values/policy/grid lines
    drawCellStates(gridSize, cellSize);

    // 4. Draw Agent - On top of items
    drawAgent(agentPos, cellSize, isAnimating ? visualAgentPos : null);

    // 5. Draw Floating Reward Text - On top of agent
    if (rewardAnimation.alpha > 0 && rewardAnimation.pos) {
        drawRewardText(ctx, rewardAnimation.text, rewardAnimation.pos, cellSize, rewardAnimation.alpha, rewardAnimation.offsetY);
    }

    // 6. Update Info Displays (does not affect canvas)
    updateQValueOrPreferenceDisplay();
    updateActionProbabilityDisplay();
    qValueDisplayDiv.style.display = '';
}

// --- Update Q-Value / Preference Display ---
function updateQValueOrPreferenceDisplay() {
    const currentState = `${agentPos.x},${agentPos.y}`;

    // Update header text based on algorithm
    if (qValueDisplayHeader) {
        const algorithm = getSelectedAlgorithm();
        if (algorithm === 'actor-critic') {
            qValueDisplayHeader.textContent = 'Action Preferences h(s,a)';
        } else if (algorithm === 'sr') {
             qValueDisplayHeader.textContent = 'Estimated Q(s,a) [from SR]';
        } else {
            qValueDisplayHeader.textContent = 'Action Values Q(s,a)';
        }
    }

    const setDisplayValue = (spanElement, value) => {
        const numericValue = value !== undefined ? value : 0;
        spanElement.textContent = numericValue.toFixed(2);
        // Use the same coloring logic for Q and H values for now
        if (numericValue > 0) {
            spanElement.style.color = 'var(--color-q-value-pos)';
        } else if (numericValue < 0) {
            spanElement.style.color = 'var(--color-q-value-neg)';
        } else {
            spanElement.style.color = 'var(--color-q-value-zero)';
        }
    };

    if (getSelectedAlgorithm() === 'actor-critic') {
        // Display H-values (preferences)
        const statePreferences = hTable[currentState] || {};
        setDisplayValue(qUpSpan, statePreferences['up']);
        setDisplayValue(qDownSpan, statePreferences['down']);
        setDisplayValue(qLeftSpan, statePreferences['left']);
        setDisplayValue(qRightSpan, statePreferences['right']);
    } else if (getSelectedAlgorithm() === 'sr') {
        // Display SR-based Q-values
        setDisplayValue(qUpSpan, calculateQValueSR(currentState, 'up', mTable, wTable, gridSize, getDiscountFactor(), takeAction, agentPos));
        setDisplayValue(qDownSpan, calculateQValueSR(currentState, 'down', mTable, wTable, gridSize, getDiscountFactor(), takeAction, agentPos));
        setDisplayValue(qLeftSpan, calculateQValueSR(currentState, 'left', mTable, wTable, gridSize, getDiscountFactor(), takeAction, agentPos));
        setDisplayValue(qRightSpan, calculateQValueSR(currentState, 'right', mTable, wTable, gridSize, getDiscountFactor(), takeAction, agentPos));
    } else {
        // Display standard Q-values
        const stateQValues = qTable[currentState] || {};
        setDisplayValue(qUpSpan, stateQValues['up']);
        setDisplayValue(qDownSpan, stateQValues['down']);
        setDisplayValue(qLeftSpan, stateQValues['left']);
        setDisplayValue(qRightSpan, stateQValues['right']);
    }
}

function updateActionProbabilityDisplay() {
     const currentState = `${agentPos.x},${agentPos.y}`;
     const actionProbs = getActionProbabilities(currentState, takeAction, agentPos);

     const setProbValue = (spanElement, value) => {
          const numericValue = value !== undefined ? value : 0.25; // Default to 0.25 if undefined
          spanElement.textContent = numericValue.toFixed(2);
          spanElement.style.color = interpolateProbColor(numericValue);
     };

     setProbValue(pUpSpan, actionProbs['up']);
     setProbValue(pDownSpan, actionProbs['down']);
     setProbValue(pLeftSpan, actionProbs['left']);
     setProbValue(pRightSpan, actionProbs['right']);
}

// --- Animation Function ---
function animateMove(startPos, endPos, duration, onComplete) {
    if (isAnimating) return;

    isAnimating = true;
    visualAgentPos = { ...startPos };
    let startTime = null;

    const step = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const progress = Math.min(1, elapsed / duration);

        visualAgentPos.x = startPos.x + (endPos.x - startPos.x) * progress;
        visualAgentPos.y = startPos.y + (endPos.y - startPos.y) * progress;

        drawEverything();

        if (progress < 1) {
            animationFrameId = requestAnimationFrame(step);
        } else {
            isAnimating = false;
            visualAgentPos = { ...endPos };
            // UPDATE CANONICAL STATE: Update environment's agentPos *after* animation
            setAgentPos(endPos);
            drawEverything();
            if (onComplete) {
                onComplete();
            }
        }
    };

    animationFrameId = requestAnimationFrame(step);
}

// --- Reward Text Animation ---
function startRewardAnimation(reward, position) {
    if (reward === 0) return;

    if (rewardAnimationFrameId) {
        cancelAnimationFrame(rewardAnimationFrameId);
    }

    rewardAnimation.text = reward > 0 ? `+${reward.toFixed(1)}` : `${reward.toFixed(1)}`;
    rewardAnimation.pos = { ...position };
    rewardAnimation.alpha = 1.0;
    rewardAnimation.offsetY = 0;
    rewardAnimation.startTime = performance.now();

    const animate = (timestamp) => {
        const elapsed = timestamp - rewardAnimation.startTime;
        const progress = Math.min(1, elapsed / rewardAnimation.duration);

        rewardAnimation.alpha = 1 - progress;
        rewardAnimation.offsetY = progress * (cellSize / 2);

        drawEverything();

        if (progress < 1) {
            rewardAnimationFrameId = requestAnimationFrame(animate);
        } else {
            rewardAnimation.alpha = 0;
            rewardAnimationFrameId = null;
            drawEverything();
        }
    };

    rewardAnimationFrameId = requestAnimationFrame(animate);
}

// --- Reward Chart Functions ---

function initializeRewardChart() {
    if (!rewardChartCanvas) {
        console.error("Reward chart canvas not found!");
        return;
    }
    rewardChartCtx = rewardChartCanvas.getContext('2d');

    if (rewardChartInstance) {
        rewardChartInstance.destroy();
    }

    const chartFontSize = 14;
    // Get computed styles based on the CURRENT theme at initialization
    const computedStyle = getComputedStyle(document.documentElement);
    // const isDarkMode = document.body.classList.contains('dark-mode'); // Check class directly
    const isDarkMode = currentTheme === 'dark'; // Use the variable set during initialization
    const gridColor = computedStyle.getPropertyValue('--color-chart-grid-line').trim();
    const labelColor = computedStyle.getPropertyValue('--color-chart-axis-label').trim();
    const tooltipBg = computedStyle.getPropertyValue('--color-chart-tooltip-bg').trim();
    const tooltipText = computedStyle.getPropertyValue('--color-chart-tooltip-text').trim();
    const smoothedLineColor = computedStyle.getPropertyValue('--color-chart-smoothed-line').trim();
    const smoothedBgColor = computedStyle.getPropertyValue('--color-chart-smoothed-bg').trim();
    const rawLineColor = computedStyle.getPropertyValue('--color-chart-raw-line').trim();
    const rawBgColor = computedStyle.getPropertyValue('--color-chart-raw-bg').trim();


    rewardChartInstance = new Chart(rewardChartCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: `Smoothed Reward (Avg over ${MOVING_AVERAGE_WINDOW} episodes)`,
                    data: [],
                    borderColor: smoothedLineColor, // Use computed value
                    backgroundColor: smoothedBgColor, // Use computed value
                    tension: 0.1,
                    pointRadius: 1,
                    borderWidth: 1.5,
                    order: 1
                },
                { // Dataset for raw episodic reward
                    label: 'Raw Episodic Reward',
                    data: [],
                    borderColor: rawLineColor, // Use computed value
                    backgroundColor: rawBgColor, // Use computed value
                    tension: 0.1,
                    pointRadius: 1.5,
                    borderWidth: 1,
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 150
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Episode',
                        color: labelColor, // Use computed value
                        font: {
                            size: chartFontSize,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                         maxTicksLimit: 15,
                         color: labelColor, // Use computed value
                         font: {
                            size: chartFontSize - 2
                         }
                    },
                    grid: {
                        color: gridColor // Use computed value
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Total Reward',
                        color: labelColor, // Use computed value
                        font: {
                            size: chartFontSize,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        color: labelColor, // Use computed value
                        font: {
                           size: chartFontSize - 2
                        }
                   },
                    grid: {
                        color: gridColor // Use computed value
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: labelColor, // Use computed value
                        font: {
                            size: chartFontSize
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: tooltipBg, // Use computed value
                    titleColor: tooltipText, // Use computed value
                    bodyColor: tooltipText, // Use computed value
                    titleFont: {
                        size: chartFontSize
                    },
                    bodyFont: {
                        size: chartFontSize - 1
                    }
                }
            }
        }
    });
}

// Helper to calculate moving average
function calculateMovingAverage(data, windowSize) {
    if (data.length === 0) return 0;
    const startIndex = Math.max(0, data.length - windowSize);
    const relevantData = data.slice(startIndex);
    const sum = relevantData.reduce((acc, val) => acc + val, 0);
    return sum / relevantData.length;
}

function updateRewardChart() {
    if (!rewardChartInstance) return;

    const chart = rewardChartInstance;
    const labels = chart.data.labels;
    const smoothedData = chart.data.datasets[0].data;
    const rawData = chart.data.datasets[1].data;

    labels.push(episodeCounter);
    smoothedData.push(smoothedEpisodicRewards[smoothedEpisodicRewards.length - 1]);
    rawData.push(episodicRewards[episodicRewards.length - 1]);

    if (labels.length > MAX_CHART_POINTS) {
        labels.shift();
        smoothedData.shift();
        rawData.shift();
    }

    chart.update();
}
// --- End Reward Chart Functions ---

// --- Learning Loop ---
function learningLoopStep() {
    if (isAnimating) return;

    currentEpisodeSteps++;
    const oldPos = { ...agentPos };

    const result = learningStep(oldPos, gridSize, takeAction, resetAgent);

    if (result.needsStop) {
        stopLearning();
        console.error("Learning stopped due to an error in algorithm selection.");
        return;
    }

    const newAgentPos = result.newAgentPos;
    const rewardReceived = result.reward;

    totalRewardForEpisode += rewardReceived;

    // Start reward text animation *before* agent moves
    if (rewardReceived !== 0) {
        // Use stepPenalty value for animation only if it's not a terminal reward
        const isTerminalReward = Math.abs(rewardReceived) >= 1; // Assuming terminal rewards >= 1 or <= -1
        const rewardToAnimate = isTerminalReward ? rewardReceived : rewardReceived;
        startRewardAnimation(rewardToAnimate, oldPos);
    }

    const episodeEndedNaturally = result.done;
    const maxStepsReached = currentEpisodeSteps >= maxStepsPerEpisode;
    const episodeEnded = episodeEndedNaturally || maxStepsReached;

    if (maxStepsReached && !episodeEndedNaturally) {
        // console.log(`Episode terminated at max steps (${maxStepsPerEpisode})`);
    }

    const afterStepLogic = () => {
        if (episodeEnded) {
            // --- Episode End Logic ---
            episodeCounter++;
            episodicRewards.push(totalRewardForEpisode);

            const smoothedReward = calculateMovingAverage(episodicRewards, MOVING_AVERAGE_WINDOW);
            smoothedEpisodicRewards.push(smoothedReward);

            updateRewardChart();

            totalRewardForEpisode = 0;
            currentEpisodeSteps = 0;

            if (getSelectedAlgorithm() === 'monte-carlo') {
                applyMonteCarloUpdates();
            }
            resetAgent();
            visualAgentPos = { ...agentPos };
            drawEverything();
            // --- End Episode End Logic ---

        } else {
             // Update displays after each step if episode not ended
             updateQValueOrPreferenceDisplay();
             updateActionProbabilityDisplay();
        }
    };

    if (oldPos.x !== newAgentPos.x || oldPos.y !== newAgentPos.y) {
        animateMove(oldPos, newAgentPos, animationDuration, afterStepLogic);
    } else {
        visualAgentPos = { ...agentPos };
        drawEverything();
        afterStepLogic();
    }
}

// --- Start/Stop/Reset Functions ---
function startLearning() {
    if (!isLearning) {
        isLearning = true;
        if (learningInterval) clearInterval(learningInterval);
        learningInterval = setInterval(learningLoopStep, simulationSpeed);
        startButton.disabled = true;
        stopButton.disabled = false;
        resetAgentButton.disabled = true;
        resetEnvironmentButton.disabled = true;
        gridSizeSlider.disabled = true;
        algorithmSelect.disabled = true;
        terminateOnRewardCheckbox.disabled = true;
    }
}

function stopLearning() {
    if (isLearning) {
        clearInterval(learningInterval);
        learningInterval = null;
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
        if (rewardAnimationFrameId) cancelAnimationFrame(rewardAnimationFrameId);
        rewardAnimationFrameId = null;
        isLearning = false;
        isAnimating = false;

        visualAgentPos = { ...agentPos };

        startButton.disabled = false;
        stopButton.disabled = true;
        resetAgentButton.disabled = false;
        resetEnvironmentButton.disabled = false;
        gridSizeSlider.disabled = false;
        algorithmSelect.disabled = false;
        terminateOnRewardCheckbox.disabled = false;
        drawEverything();

        if (rewardAnimationFrameId) cancelAnimationFrame(rewardAnimationFrameId);
        rewardAnimation = { text: '', pos: null, alpha: 0, offsetY: 0, startTime: 0, duration: 600 };
        rewardAnimationFrameId = null;

        // Update UI elements using getter functions...
        lrValueSpan.textContent = getLearningRate().toFixed(2);
        discountValueSpan.textContent = getDiscountFactor().toFixed(2);
        epsilonValueSpan.textContent = getExplorationRate().toFixed(2);
        softmaxBetaValueSpan.textContent = getSoftmaxBeta().toFixed(1);
        speedValueSpan.textContent = (1010 - simulationSpeed).toString();
        gridSizeValueSpan.textContent = gridSizeSlider.value;
        stepPenaltyValueSpan.textContent = parseFloat(stepPenaltySlider.value).toFixed(1);
        maxStepsValueSpan.textContent = maxStepsSlider.value;
        gemRewardValueSpan.textContent = gemRewardSlider.value;
        badStateRewardValueSpan.textContent = badStateRewardSlider.value;

        lrSlider.value = getLearningRate();
        discountSlider.value = getDiscountFactor();
        epsilonSlider.value = getExplorationRate();
        softmaxBetaSlider.value = getSoftmaxBeta();
        gridSizeSlider.value = gridSize;
        maxStepsSlider.value = maxStepsPerEpisode;
        algorithmSelect.value = getSelectedAlgorithm();
        speedSlider.value = 1010 - simulationSpeed;
        cellDisplayMode = cellDisplayModeSelect.value;
        gemRewardSlider.value = parseFloat(gemRewardValueSpan.textContent);
        badStateRewardSlider.value = parseFloat(badStateRewardValueSpan.textContent);

        updateSpeed();
    }
}

// Add a base reset function:
function performReset(resetType) {
    stopLearning();
    
    // Common reset actions
    if (resetType !== 'agent-only') {
        updateTerminateOnGemSetting();
        updateSelectedAlgorithm(algorithmSelect.value);
        maxStepsPerEpisode = parseInt(maxStepsSlider.value, 10);
        setStepPenalty(parseFloat(stepPenaltySlider.value));
    }
    
    // Environment reset
    if (resetType === 'full' || resetType === 'environment') {
        initializeGridRewards(gridSize);
        setStartPos({ x: 0, y: 0 }, gridSize);
    }
    
    // Agent knowledge reset
    if (resetType === 'full' || resetType === 'agent') {
        initializeTables(gridSize);
        
        // Reset chart data and counters
        episodeCounter = 0;
        totalRewardForEpisode = 0;
        episodicRewards = [];
        smoothedEpisodicRewards = [];
        episodeNumbers = [];
        initializeRewardChart();
    }
    
    // Physical agent reset (always needed)
    resetAgent();
    
    // Visual state reset
    visualAgentPos = { ...agentPos };
    isAnimating = false;
    currentEpisodeSteps = 0;
    
    if (resetType === 'full') {
        updateExplanationText();
    }
    
    drawEverything();
}

// Simplify the individual reset functions:
function resetAllAndDraw() {
    performReset('full');
}

function resetAgentLogic() {
    performReset('agent');
    console.log("Agent learning progress (Q/V/H tables) reset.");
}

function resetEnvironmentLogic() {
    performReset('environment');
    console.log("Environment layout and start position reset to default.");
}

// --- UI Update Functions ---
function updateTerminateOnGemSetting() {
    terminateOnGem = terminateOnRewardCheckbox.checked;
    setTerminateOnGem(terminateOnGem); // Update the setting in environment.js
    console.log("Terminate on Gem:", terminateOnGem);
}

function updateSpeed() {
    const sliderValue = parseInt(speedSlider.value, 10);
    const minDelay = 10;
    const maxDelay = 1000;
    // Invert slider value: min slider value (10) -> max delay (1000), max slider value (1000) -> min delay (10)
    simulationSpeed = maxDelay + minDelay - sliderValue;
    // Adjust animation duration based on speed, but keep it reasonably fast
    animationDuration = Math.min(simulationSpeed * 0.8, 150); // e.g., 80% of step time, max 150ms

    speedValueSpan.textContent = sliderValue.toString();

    if (isLearning) {
        clearInterval(learningInterval);
        learningInterval = setInterval(learningLoopStep, simulationSpeed);
    }
}
// --- END UI Update Functions ---

// --- Function to load explanations ---
async function loadExplanations() {
    try {
        const response = await fetch('./explanations.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        explanations = await response.json();
        console.log("Explanations loaded successfully.");
        updateExplanationText();
    } catch (error) {
        console.error("Could not load explanations:", error);
        // Handle error: display an error message in the UI?
        explanationTitle.textContent = 'Error Loading Explanations';
        algorithmExplanationDiv.innerHTML = '<p>Could not load explanation content. Please check the console.</p>';
        explorationExplanationDiv.innerHTML = '';
    }
}

// --- NEW: URL-based config persistence helpers ---
function getCurrentConfigParams() {
    const params = new URLSearchParams();
    params.set('algo', getSelectedAlgorithm());
    params.set('strategy', getExplorationStrategy());
    params.set('lr', getLearningRate().toString());
    params.set('discount', getDiscountFactor().toString());
    params.set('epsilon', getExplorationRate().toString());
    params.set('beta', getSoftmaxBeta().toString());
    params.set('actorLr', getActorLearningRate().toString());
    params.set('criticLr', getCriticLearningRate().toString());
    params.set('srMLr', getSRMWeightLearningRate().toString());
    params.set('srWLr', getSRWWeightLearningRate().toString());
    params.set('grid', gridSize.toString());
    params.set('display', cellDisplayMode);
    params.set('stepPenalty', stepPenaltySlider.value);
    params.set('gemReward', gemRewardSlider.value);
    params.set('badReward', badStateRewardSlider.value);
    params.set('maxSteps', maxStepsPerEpisode.toString());
    params.set('speed', speedSlider.value);
    params.set('terminate', terminateOnGem ? '1' : '0');
    return params;
}
function updateURL() {
    const qs = getCurrentConfigParams().toString();
    history.replaceState(null, '', window.location.pathname + (qs ? `?${qs}` : ''));
}

// --- Event Listeners ---
startButton.addEventListener('click', startLearning);
stopButton.addEventListener('click', stopLearning);
resetAgentButton.addEventListener('click', resetAgentLogic);
resetEnvironmentButton.addEventListener('click', resetEnvironmentLogic);

// Add this helper function near the top:
function createSliderHandler(updateFunction, valueSpan, formatter = v => v.toFixed(2)) {
    return (slider) => {
        slider.addEventListener('input', () => {
            const value = parseFloat(slider.value);
            updateFunction(value);
            valueSpan.textContent = formatter(value);
            updateURL(); // ← push into URL whenever a slider moves
        });
    };
}

// Replace repetitive slider event listeners:
createSliderHandler(updateLearningRate, lrValueSpan)(lrSlider);
createSliderHandler(updateDiscountFactor, discountValueSpan)(discountSlider);
createSliderHandler(updateExplorationRate, epsilonValueSpan)(epsilonSlider);
createSliderHandler(updateSoftmaxBeta, softmaxBetaValueSpan, (v) => v.toFixed(1))(softmaxBetaSlider);
createSliderHandler(setStepPenalty, stepPenaltyValueSpan, (v) => v.toFixed(1))(stepPenaltySlider);
createSliderHandler(setGemRewardMagnitude, gemRewardValueSpan, (v) => v.toString())(gemRewardSlider);
createSliderHandler(setBadStateRewardMagnitude, badStateRewardValueSpan, (v) => v.toString())(badStateRewardSlider);

gridSizeSlider.addEventListener('input', () => {
    const newSizeValue = parseInt(gridSizeSlider.value, 10);
    gridSizeValueSpan.textContent = newSizeValue.toString();

    // Pass a temporary object mimicking the old input structure to updateGridSize
    const fakeInput = { value: newSizeValue.toString() };
    const { gridSize: newGridSize, cellSize: newCellSize, updated } = updateGridSize(fakeInput, gridSize);

    if (updated) {
        gridSize = newGridSize;
        cellSize = newCellSize;

        // A grid size change requires a full reset of environment AND agent knowledge
        resetAllAndDraw();

        console.log("Grid size changed, full reset performed.");
    }
    updateURL(); // ← record it
});

// Add this helper function:
function updateControlVisibility(algorithm, explorationStrategy) {
    const controls = {
        strategy: explorationStrategySelect.parentElement,
        epsilon: epsilonSlider.parentElement.parentElement,
        softmaxBeta: softmaxBetaControl,
        mainLR: lrControl,
        actorLR: actorCriticLRControl,
        criticLR: criticLRControl,
        srMLR: srMLRControl,
        srWLR: srWLRControl,
        srVectorAgent: srVectorAgentDisplayOption,
        srVectorHover: srVectorHoverDisplayOption,
        srWVector: srWVectorDisplayOption // NEW: Add W vector control
    };

    // Hide all controls initially
    Object.values(controls).forEach(control => {
        if (control) control.style.display = 'none';
    });

    // Show controls based on algorithm
    if (algorithm === 'actor-critic') {
        controls.softmaxBeta.style.display = '';
        controls.actorLR.style.display = '';
        controls.criticLR.style.display = '';
    } else if (algorithm === 'sr') {
        controls.strategy.style.display = '';
        controls.srMLR.style.display = '';
        controls.srWLR.style.display = '';
        controls.srVectorAgent.style.display = '';
        controls.srVectorHover.style.display = '';
        controls.srWVector.style.display = ''; // NEW: Show W vector option for SR
        updateExplorationControlVisibility(explorationStrategy);
    } else {
        controls.strategy.style.display = '';
        controls.mainLR.style.display = '';
        updateExplorationControlVisibility(explorationStrategy);
    }
}

// Use this function in algorithmSelect event listener and initializeApp:
algorithmSelect.addEventListener('change', () => {
    stopLearning();
    const newAlgo = algorithmSelect.value;
    updateSelectedAlgorithm(newAlgo);
    updateExplanationText();
    updateControlVisibility(newAlgo, explorationStrategySelect.value);
    if (!['sr'].includes(newAlgo) && ['sr-vector','sr-vector-hover','sr-w-vector'].includes(cellDisplayModeSelect.value)) {
        cellDisplayModeSelect.value = 'values-color';
        cellDisplayMode = 'values-color';
    }
    resetAgentLogic();
    console.log("Algorithm changed to:", newAlgo);
    updateURL(); // ← record it
});

// Helper function to update exploration control visibility
function updateExplorationControlVisibility(strategy) {
    const epsilonField = epsilonSlider.parentElement.parentElement;
    if (strategy === 'epsilon-greedy') {
        epsilonField.style.display = '';
        softmaxBetaControl.style.display = 'none';
    } else if (strategy === 'softmax') {
        epsilonField.style.display = 'none';
        softmaxBetaControl.style.display = '';
    } else { // Handles 'random' and 'greedy'
        epsilonField.style.display = 'none';
        softmaxBetaControl.style.display = 'none';
    }
}

explorationStrategySelect.addEventListener('change', () => {
    if (getSelectedAlgorithm() !== 'actor-critic') {
        stopLearning();
        const strat = explorationStrategySelect.value;
        updateExplorationStrategy(strat);
        updateExplanationText();
        updateExplorationControlVisibility(strat);
        drawEverything();
        console.log("Exploration strategy changed to:", strat);
        updateURL(); // ← record it
    }
});

terminateOnRewardCheckbox.addEventListener('change', () => {
    updateTerminateOnGemSetting();
    updateURL(); // ← record it
});
speedSlider.addEventListener('input', () => {
    updateSpeed();
    updateURL(); // ← record it
});

// REMOVED: Allow clicking while learning
canvas.addEventListener('click', (event) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const gridX = Math.floor(mouseX / cellSize);
    const gridY = Math.floor(mouseY / cellSize);

    if (event.shiftKey) {
        // --- Set Start Position Logic ---
        console.log(`Shift+Click detected on cell (${gridX}, ${gridY})`);
        if (setStartPos({ x: gridX, y: gridY }, gridSize)) {
             // 1. Reset the agent *if* learning is not active, to move it immediately.
             //    If learning is active, it will move on the next episode end/reset.
             if (!isLearning) {
                resetAgent();
                visualAgentPos = { ...agentPos };
             }
             // 2. Redraw everything to show the updated home icon and potentially moved agent.
             drawEverything();
        } else {
            // Optional: Add visual feedback if setting start pos failed (e.g., flash cell red)
             console.warn("Failed to set start position here.");
        }
    } else {
        // --- Cycle Cell State Logic (Original) ---
        if (cycleCellState(gridX, gridY, gridSize)) {
            drawEverything();
            // Let the agent learn the new state dynamically.
        }
    }
});

// Listener for Cell Display Mode Select
cellDisplayModeSelect.addEventListener('change', () => {
    cellDisplayMode = cellDisplayModeSelect.value;
    console.log("Cell display mode changed to:", cellDisplayMode);
    drawEverything();
    updateURL(); // ← record it
});

// Event listener for Max Steps slider
maxStepsSlider.addEventListener('input', () => {
    const value = parseInt(maxStepsSlider.value, 10);
    maxStepsPerEpisode = value;
    maxStepsValueSpan.textContent = value.toString();
    updateURL(); // ← record it
});

// --- NEW: Theme Switching Logic ---
function setTheme(theme) {
    if (theme === 'dark') {
        document.body.classList.add('dark-mode');
        themeToggleCheckbox.checked = false; // Dark mode = unchecked (moon visible)
        currentTheme = 'dark';
    } else {
        document.body.classList.remove('dark-mode');
        themeToggleCheckbox.checked = true; // Light mode = checked (sun visible)
        currentTheme = 'light';
    }
    localStorage.setItem('theme', theme); // Save preference

    // Update Chart.js colors AFTER the theme class has been applied to the body
    updateChartTheme();
}

function updateChartTheme() {
    if (!rewardChartInstance) return;

    // Get computed styles AFTER the theme class has been updated
    const computedStyle = getComputedStyle(document.documentElement);
    const gridColor = computedStyle.getPropertyValue('--color-chart-grid-line').trim();
    const labelColor = computedStyle.getPropertyValue('--color-chart-axis-label').trim(); // Used for axis ticks, titles, and legend
    const tooltipBg = computedStyle.getPropertyValue('--color-chart-tooltip-bg').trim();
    const tooltipText = computedStyle.getPropertyValue('--color-chart-tooltip-text').trim(); // Used for tooltip title and body
    const smoothedLineColor = computedStyle.getPropertyValue('--color-chart-smoothed-line').trim();
    const smoothedBgColor = computedStyle.getPropertyValue('--color-chart-smoothed-bg').trim();
    const rawLineColor = computedStyle.getPropertyValue('--color-chart-raw-line').trim();
    const rawBgColor = computedStyle.getPropertyValue('--color-chart-raw-bg').trim();

    // Update chart options
    const options = rewardChartInstance.options;

    // --- Update Text Colors ---
    // Scales (Axes Ticks and Titles)
    options.scales.x.ticks.color = labelColor;
    options.scales.y.ticks.color = labelColor;
    options.scales.x.title.color = labelColor; // Explicitly update title color
    options.scales.y.title.color = labelColor; // Explicitly update title color

    // Grid lines
    options.scales.x.grid.color = gridColor;
    options.scales.y.grid.color = gridColor;

    // Plugins (Legend and Tooltip)
    if (options.plugins.legend && options.plugins.legend.labels) {
        options.plugins.legend.labels.color = labelColor; // Explicitly update legend label color
    }
    if (options.plugins.tooltip) {
        options.plugins.tooltip.backgroundColor = tooltipBg;
        options.plugins.tooltip.titleColor = tooltipText; // Explicitly update tooltip title color
        options.plugins.tooltip.bodyColor = tooltipText;  // Explicitly update tooltip body color
    }
    // --- End Update Text Colors ---


    // Update dataset colors (Line and Background)
    const datasets = rewardChartInstance.data.datasets;
    datasets[0].borderColor = smoothedLineColor;
    datasets[0].backgroundColor = smoothedBgColor;
    datasets[1].borderColor = rawLineColor;
    datasets[1].backgroundColor = rawBgColor;


    rewardChartInstance.update('none'); // Update chart without animation to avoid jarring changes
}

// --- End Theme Switching Logic ---

// NEW: Theme Toggle Listener
themeToggleCheckbox.addEventListener('change', () => {
    setTheme(themeToggleCheckbox.checked ? 'light' : 'dark');
});

// --- Initial Setup ---
async function initializeApp() {
    // NEW: read URL params and re-apply
    const urlParams = new URLSearchParams(window.location.search);

    if (urlParams.has('algo')) {
        const a = urlParams.get('algo');
        if ([ 'q-learning','sarsa','expected-sarsa','monte-carlo','actor-critic','sr' ].includes(a)) {
            algorithmSelect.value = a;
            updateSelectedAlgorithm(a);
        }
    }
    if (urlParams.has('strategy')) {
        const s = urlParams.get('strategy');
        if ([ 'epsilon-greedy','softmax','random','greedy' ].includes(s)) {
            explorationStrategySelect.value = s;
            updateExplorationStrategy(s);
        }
    }
    if (urlParams.has('lr')) {
        const v = parseFloat(urlParams.get('lr'));
        if (!isNaN(v) && lrSlider && lrValueSpan) { updateLearningRate(v); lrSlider.value = v; lrValueSpan.textContent = v.toFixed(2); }
    }
    if (urlParams.has('discount')) {
        const v = parseFloat(urlParams.get('discount'));
        if (!isNaN(v) && discountSlider && discountValueSpan) { updateDiscountFactor(v); discountSlider.value = v; discountValueSpan.textContent = v.toFixed(2); }
    }
    if (urlParams.has('epsilon')) {
        const v = parseFloat(urlParams.get('epsilon'));
        if (!isNaN(v) && epsilonSlider && epsilonValueSpan) { updateExplorationRate(v); epsilonSlider.value = v; epsilonValueSpan.textContent = v.toFixed(2); }
    }
    if (urlParams.has('beta')) {
        const v = parseFloat(urlParams.get('beta'));
        if (!isNaN(v) && softmaxBetaSlider && softmaxBetaValueSpan) { updateSoftmaxBeta(v); softmaxBetaSlider.value = v; softmaxBetaValueSpan.textContent = v.toFixed(1); }
    }
    if (urlParams.has('actorLr')) {
        const v = parseFloat(urlParams.get('actorLr'));
        if (!isNaN(v) && actorLrSlider && actorLrValueSpan) { // Check elements
            updateActorLearningRate(v); 
            actorLrSlider.value = v; 
            actorLrValueSpan.textContent = v.toFixed(2); 
        }
    }
    if (urlParams.has('criticLr')) {
        const v = parseFloat(urlParams.get('criticLr'));
        if (!isNaN(v) && criticLrSlider && criticLrValueSpan) { // Check elements
            updateCriticLearningRate(v); 
            criticLrSlider.value = v; 
            criticLrValueSpan.textContent = v.toFixed(2); 
        }
    }
    if (urlParams.has('srMLr')) {
        const v = parseFloat(urlParams.get('srMLr'));
        if (!isNaN(v) && srMLrSlider && srMLrValueSpan) { // Check elements
            updateSRMLearningRate(v); 
            srMLrSlider.value = v; 
            srMLrValueSpan.textContent = v.toFixed(2); 
        }
    }
    if (urlParams.has('srWLr')) {
        const v = parseFloat(urlParams.get('srWLr'));
        if (!isNaN(v) && srWLrSlider && srWLrValueSpan) { // Check elements
            updateSRWLearningRate(v); 
            srWLrSlider.value = v; 
            srWLrValueSpan.textContent = v.toFixed(2); 
        }
    }
    if (urlParams.has('grid')) {
        const g = parseInt(urlParams.get('grid'),10);
        if (!isNaN(g) && gridSizeSlider && gridSizeValueSpan) {
            const fake = { value: g.toString() };
            const { gridSize: newG, cellSize: newC, updated } = updateGridSize(fake, gridSize); // ensure 'updated' is handled if necessary, though resetAllAndDraw later will use the new gridSize
            gridSize = newG; cellSize = newC; // Directly update global gridSize and cellSize
            gridSizeSlider.value = newG; gridSizeValueSpan.textContent = newG.toString();
            // If updateGridSize itself doesn't call reset/redraw, ensure it happens. 
            // resetAllAndDraw() at the end of initializeApp will handle the full reset based on this new grid size.
        }
    }
    if (urlParams.has('display')) {
        const d = urlParams.get('display');
        if (cellDisplayModeSelect && Array.from(cellDisplayModeSelect.options).some(o=>o.value===d)) {
            cellDisplayModeSelect.value = d;
            cellDisplayMode = d;
        }
    }
    if (urlParams.has('stepPenalty')) {
        const v = parseFloat(urlParams.get('stepPenalty'));
        if (!isNaN(v) && stepPenaltySlider && stepPenaltyValueSpan) { setStepPenalty(v); stepPenaltySlider.value = v; stepPenaltyValueSpan.textContent = v.toFixed(1); }
    }
    if (urlParams.has('gemReward')) {
        const v = parseFloat(urlParams.get('gemReward'));
        if (!isNaN(v) && gemRewardSlider && gemRewardValueSpan) { setGemRewardMagnitude(v); gemRewardSlider.value = v; gemRewardValueSpan.textContent = v.toString(); }
    }
    if (urlParams.has('badReward')) {
        const v = parseFloat(urlParams.get('badReward'));
        if (!isNaN(v) && badStateRewardSlider && badStateRewardValueSpan) { setBadStateRewardMagnitude(v); badStateRewardSlider.value = v; badStateRewardValueSpan.textContent = v.toString(); }
    }
    if (urlParams.has('maxSteps')) {
        const v = parseInt(urlParams.get('maxSteps'),10);
        if (!isNaN(v) && maxStepsSlider && maxStepsValueSpan) { maxStepsPerEpisode = v; maxStepsSlider.value = v; maxStepsValueSpan.textContent = v.toString(); }
    }
    if (urlParams.has('speed')) {
        const v = parseInt(urlParams.get('speed'),10);
        if (!isNaN(v) && speedSlider) { speedSlider.value = v; updateSpeed(); }
    }
    if (urlParams.has('terminate')) {
        const t = urlParams.get('terminate');
        const b = (t==='1' || t==='true'); // Handles '1' or 'true' as true
        if (terminateOnRewardCheckbox) terminateOnRewardCheckbox.checked = b;
        // setTheme; // This line seems incomplete/incorrect from the previous diff, should be removed or be setTheme(themeValue) if theme is in URL
        updateTerminateOnGemSetting(); // This will read from the checkbox
    }
    // Now fix your control-visibility and explanations:
    updateControlVisibility(getSelectedAlgorithm(), getExplorationStrategy()); // Ensure this is called after all params are set
    updateExplanationText(); // Ensure this is called after algo/strategy are set
    // ... then continue with initializeCollapsibles() as before ...

    initializeCollapsibles();

    const onAppReady = () => {
        resetAllAndDraw(); // Draw initial state *after* everything is ready
        // Set button states after reset
        stopButton.disabled = true;
        resetAgentButton.disabled = false;
        resetEnvironmentButton.disabled = false;
        startButton.disabled = false;
        console.log("App initialized and ready.");
    };

    // Use Promise.all to wait for both explanations and images to load
    try {
        // Load images returns a promise now, call it directly
        await Promise.all([
            loadExplanations(),
            loadImages() // Call loadImages() here
        ]);
        // Now both are loaded, proceed with drawing etc.
        onAppReady();
    } catch (error) {
        console.error("Error during initialization:", error);
        // Handle initialization error (e.g., display error message)
        // You might want to show an error to the user here
        explanationTitle.textContent = 'Initialization Error';
        algorithmExplanationDiv.innerHTML = `<p>Error loading app assets: ${error.message}. Please check the console and refresh.</p>`;
    }
}

// Start the application initialization
initializeApp(); 