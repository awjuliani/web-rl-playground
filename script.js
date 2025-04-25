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
    rewardMagnitudeBad as initialBadReward // NEW: Import initial bad reward
} from './environment.js';

import {
    qTable, // Import qTable directly from algorithms.js
    vTable, // Import vTable
    hTable, // Import hTable
    mTable, // Import mTable
    wTable, // Import wTable
    // Import CURRENT parameter values
    learningRate, discountFactor, explorationRate, softmaxBeta,
    explorationStrategy, selectedAlgorithm,
    // Keep initial values for initial setup
    learningRate as initialLr, discountFactor as initialDf, explorationRate as initialEr,
    softmaxBeta as initialBeta, explorationStrategy as initialExplorationStrategy,
    selectedAlgorithm as initialAlgo,
    initializeTables, // Use the renamed function
    learningStep,
    updateLearningRate, updateDiscountFactor, updateExplorationRate, updateSoftmaxBeta, updateExplorationStrategy,
    updateSelectedAlgorithm, applyMonteCarloUpdates,
    getActionProbabilities, // Import the new function
    getBestActions,
    calculateQValueSR, // Import SR Q calculation helper
    // RENAMED: Import specific learning rates (srW instead of srR)
    actorLearningRate, criticLearningRate, srMWeightLearningRate, srWWeightLearningRate,
    // RENAMED: Import specific LR update functions (updateSRW instead of updateSRR)
    updateActorLearningRate, updateCriticLearningRate, updateSRMLearningRate, updateSRWLearningRate,
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
const qStateSpan = document.getElementById('qState');
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
const srVectorAgentDisplayOption = document.getElementById('srVectorAgentDisplayOption'); // Renamed from srVectorDisplayOption
const srVectorHoverDisplayOption = document.getElementById('srVectorHoverDisplayOption'); // NEW: Get the hover option
const themeToggleCheckbox = document.getElementById('theme-checkbox'); // NEW: Theme toggle
const gemRewardSlider = document.getElementById('gemRewardSlider'); // NEW
const gemRewardValueSpan = document.getElementById('gemRewardValue'); // NEW
const badStateRewardSlider = document.getElementById('badStateRewardSlider'); // NEW
const badStateRewardValueSpan = document.getElementById('badStateRewardValue'); // NEW
const srMLRControl = document.getElementById('srMLRControl');
const srMLrSlider = document.getElementById('srMLrSlider');
const srMLrValueSpan = document.getElementById('srMLrValue');
const srWLRControl = document.getElementById('srWLRControl');
const srWLrSlider = document.getElementById('srWLrSlider');
const srWLrValueSpan = document.getElementById('srWLrValue');

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

    const algoKey = selectedAlgorithm;
    const strategyKey = explorationStrategy;

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

    // 1. Draw Cell Background/Policy/SR Vector OR Nothing
    if (cellDisplayMode === 'values-color') {
        drawValues(ctx, gridSize, cellSize, qTable, vTable, mTable, wTable, selectedAlgorithm, false);
    } else if (cellDisplayMode === 'values-text') {
        drawValues(ctx, gridSize, cellSize, qTable, vTable, mTable, wTable, selectedAlgorithm, true);
    } else if (cellDisplayMode === 'policy') {
        drawPolicyArrows(ctx, gridSize, cellSize, qTable, hTable, mTable, wTable, selectedAlgorithm, takeAction, agentPos);
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
        if (selectedAlgorithm === 'actor-critic') {
            qValueDisplayHeader.textContent = 'Action Preferences h(s,a)';
        } else if (selectedAlgorithm === 'sr') {
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

    if (selectedAlgorithm === 'actor-critic') {
        // Display H-values (preferences)
        const statePreferences = hTable[currentState] || {};
        setDisplayValue(qUpSpan, statePreferences['up']);
        setDisplayValue(qDownSpan, statePreferences['down']);
        setDisplayValue(qLeftSpan, statePreferences['left']);
        setDisplayValue(qRightSpan, statePreferences['right']);
    } else if (selectedAlgorithm === 'sr') {
        // Display SR-based Q-values
        setDisplayValue(qUpSpan, calculateQValueSR(currentState, 'up', mTable, wTable, gridSize, discountFactor, takeAction, agentPos));
        setDisplayValue(qDownSpan, calculateQValueSR(currentState, 'down', mTable, wTable, gridSize, discountFactor, takeAction, agentPos));
        setDisplayValue(qLeftSpan, calculateQValueSR(currentState, 'left', mTable, wTable, gridSize, discountFactor, takeAction, agentPos));
        setDisplayValue(qRightSpan, calculateQValueSR(currentState, 'right', mTable, wTable, gridSize, discountFactor, takeAction, agentPos));
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

            if (selectedAlgorithm === 'monte-carlo') {
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

        // Update UI elements...
        lrValueSpan.textContent = learningRate.toFixed(2);
        discountValueSpan.textContent = discountFactor.toFixed(2);
        epsilonValueSpan.textContent = explorationRate.toFixed(2);
        softmaxBetaValueSpan.textContent = softmaxBeta.toFixed(1);
        speedValueSpan.textContent = (1010 - simulationSpeed).toString();
        gridSizeValueSpan.textContent = gridSizeSlider.value;
        stepPenaltyValueSpan.textContent = parseFloat(stepPenaltySlider.value).toFixed(1);
        maxStepsValueSpan.textContent = maxStepsSlider.value;
        gemRewardValueSpan.textContent = gemRewardSlider.value;
        badStateRewardValueSpan.textContent = badStateRewardSlider.value;

        lrSlider.value = learningRate;
        discountSlider.value = discountFactor;
        epsilonSlider.value = explorationRate;
        softmaxBetaSlider.value = softmaxBeta;
        gridSizeSlider.value = gridSize;
        maxStepsSlider.value = maxStepsPerEpisode;
        algorithmSelect.value = selectedAlgorithm;
        speedSlider.value = 1010 - simulationSpeed;
        cellDisplayMode = cellDisplayModeSelect.value;
        gemRewardSlider.value = parseFloat(gemRewardValueSpan.textContent);
        badStateRewardSlider.value = parseFloat(badStateRewardValueSpan.textContent);

        updateSpeed();
    }
}

// Renamed original reset to resetAllAndDraw - used for full resets (init, grid size change)
function resetAllAndDraw() {
    stopLearning();

    updateTerminateOnGemSetting();
    updateSelectedAlgorithm(algorithmSelect.value);
    maxStepsPerEpisode = parseInt(maxStepsSlider.value, 10);
    setStepPenalty(parseFloat(stepPenaltySlider.value));

    // Reset environment layout AND agent start position to default
    initializeGridRewards(gridSize);
    setStartPos({ x: 0, y: 0 }, gridSize);

    // Reset agent state and learning progress
    initializeTables(gridSize);
    resetAgent();

    // Reset chart data and counters
    episodeCounter = 0;
    totalRewardForEpisode = 0;
    episodicRewards = [];
    smoothedEpisodicRewards = [];
    episodeNumbers = [];
    initializeRewardChart();

    // Reset animation/visual state
    visualAgentPos = { ...agentPos };
    isAnimating = false;
    currentEpisodeSteps = 0;

    updateExplanationText();

    drawEverything();
}

// Logic for Reset Agent button
function resetAgentLogic() {
    stopLearning();

    // Reset only the agent's knowledge and progress
    initializeTables(gridSize);
    resetAgent();

    // Reset chart data and counters
    episodeCounter = 0;
    totalRewardForEpisode = 0;
    episodicRewards = [];
    smoothedEpisodicRewards = [];
    episodeNumbers = [];
    initializeRewardChart();

    // Reset animation/visual state
    visualAgentPos = { ...agentPos };
    isAnimating = false;
    currentEpisodeSteps = 0;

    drawEverything();
    console.log("Agent learning progress (Q/V/H tables) reset.");
}

// Logic for Reset Environment button
function resetEnvironmentLogic() {
    stopLearning();

    // Reset only the environment layout and agent's physical position
    initializeGridRewards(gridSize);
    setStartPos({ x: 0, y: 0 }, gridSize);
    resetAgent();

    // Reset animation/visual state (agent moved)
    visualAgentPos = { ...agentPos };
    isAnimating = false;
    currentEpisodeSteps = 0;

    drawEverything();
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

// --- Event Listeners ---
startButton.addEventListener('click', startLearning);
stopButton.addEventListener('click', stopLearning);
resetAgentButton.addEventListener('click', resetAgentLogic);
resetEnvironmentButton.addEventListener('click', resetEnvironmentLogic);

lrSlider.addEventListener('input', () => {
    const value = parseFloat(lrSlider.value);
    updateLearningRate(value);
    lrValueSpan.textContent = value.toFixed(2);

    // ALSO update the specific sliders and their display values to match the main LR
    // This provides a consistent baseline when the main slider is moved.
    actorLrSlider.value = value;
    actorLrValueSpan.textContent = value.toFixed(2);
    criticLrSlider.value = value;
    criticLrValueSpan.textContent = value.toFixed(2);
    srMLrSlider.value = value;
    srMLrValueSpan.textContent = value.toFixed(2);
    srWLrSlider.value = value;
    srWLrValueSpan.textContent = value.toFixed(2);
});

discountSlider.addEventListener('input', () => {
    const value = parseFloat(discountSlider.value);
    updateDiscountFactor(value);
    discountValueSpan.textContent = value.toFixed(2);
});
epsilonSlider.addEventListener('input', () => {
    const value = parseFloat(epsilonSlider.value);
    updateExplorationRate(value);
    epsilonValueSpan.textContent = value.toFixed(2);
});
softmaxBetaSlider.addEventListener('input', () => {
    const value = parseFloat(softmaxBetaSlider.value);
    updateSoftmaxBeta(value);
    softmaxBetaValueSpan.textContent = value.toFixed(1);
});
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
});
algorithmSelect.addEventListener('change', () => {
    stopLearning();
    const newAlgo = algorithmSelect.value;
    updateSelectedAlgorithm(newAlgo);
    updateExplanationText();

    // --- UI Element Visibility ---
    const strategyField = explorationStrategySelect.parentElement;
    const epsilonField = epsilonSlider.parentElement.parentElement;

    // Hide all specific LR controls by default
    actorCriticLRControl.style.display = 'none';
    criticLRControl.style.display = 'none';
    srMLRControl.style.display = 'none';
    srWLRControl.style.display = 'none';
    // Hide exploration strategy/epsilon/softmax controls initially, show as needed
    if (strategyField) strategyField.style.display = 'none';
    if (epsilonField) epsilonField.style.display = 'none';
    softmaxBetaControl.style.display = 'none';
    // Hide main LR slider by default, show as needed
    if (lrControl) lrControl.style.display = 'none';


    // --- SR Display Option Visibility ---
    if (newAlgo === 'sr') {
        srVectorAgentDisplayOption.style.display = '';
        srVectorHoverDisplayOption.style.display = '';
        // Show SR specific LR controls
        srMLRControl.style.display = '';
        srWLRControl.style.display = '';
        // Hide main LR control for SR
        if (lrControl) lrControl.style.display = 'none';
    } else {
        srVectorAgentDisplayOption.style.display = 'none';
        srVectorHoverDisplayOption.style.display = 'none';
         if (cellDisplayModeSelect.value === 'sr-vector' || cellDisplayModeSelect.value === 'sr-vector-hover') {
             cellDisplayModeSelect.value = 'values-color';
             cellDisplayMode = 'values-color';
         }
    }
    // --- End SR Display Option Visibility ---

    // --- Exploration & Specific/Main LR Control Visibility ---
    if (newAlgo === 'actor-critic') {
        // AC uses softmax implicitly, show Beta control
        softmaxBetaControl.style.display = '';
        // Show Actor-Critic specific LR controls
        actorCriticLRControl.style.display = '';
        criticLRControl.style.display = '';
        // Hide main LR control for AC
        if (lrControl) lrControl.style.display = 'none';
        // Hide general exploration strategy/epsilon
        if (strategyField) strategyField.style.display = 'none';
        if (epsilonField) epsilonField.style.display = 'none';
    } else if (newAlgo === 'sr') {
        // SR uses standard exploration, show strategy dropdown
        if (strategyField) strategyField.style.display = '';
         const currentStrategy = explorationStrategySelect.value;
         if (currentStrategy === 'epsilon-greedy') {
            if (epsilonField) epsilonField.style.display = '';
            softmaxBetaControl.style.display = 'none';
         } else { // Softmax
            if (epsilonField) epsilonField.style.display = 'none';
            softmaxBetaControl.style.display = '';
         }
         // SR specific LR controls (srWLRControl) are already shown above
         // Main LR control is already hidden above
    } else { // Handles QL, SARSA, ES, MC (non-AC, non-SR algos)
         // Show standard exploration strategy controls
         if (strategyField) strategyField.style.display = '';
         const currentStrategy = explorationStrategySelect.value;
         if (currentStrategy === 'epsilon-greedy') {
            if (epsilonField) epsilonField.style.display = '';
            softmaxBetaControl.style.display = 'none';
         } else { // Softmax
            if (epsilonField) epsilonField.style.display = 'none';
            softmaxBetaControl.style.display = '';
         }
         // Show main LR control for these algorithms
         if (lrControl) lrControl.style.display = '';
         // Ensure AC/SR specific LR controls remain hidden (done by default at start)
    }
    // --- End Exploration Control Visibility ---

    resetAgentLogic();

    console.log("Algorithm changed to:", newAlgo, "- Agent reset.");
});
explorationStrategySelect.addEventListener('change', () => {
    if (selectedAlgorithm !== 'actor-critic') {
        stopLearning();
        const newStrategy = explorationStrategySelect.value;
        updateExplorationStrategy(newStrategy);
        updateExplanationText();

        const epsilonField = epsilonSlider.parentElement.parentElement;

        if (newStrategy === 'epsilon-greedy') {
            epsilonField.style.display = '';
            softmaxBetaControl.style.display = 'none';
        } else if (newStrategy === 'softmax') {
            epsilonField.style.display = 'none';
            softmaxBetaControl.style.display = '';
        }
        drawEverything();
        console.log("Exploration strategy changed to:", newStrategy);
    }
});
terminateOnRewardCheckbox.addEventListener('change', updateTerminateOnGemSetting);
speedSlider.addEventListener('input', updateSpeed);

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

stepPenaltySlider.addEventListener('input', () => {
    const value = parseFloat(stepPenaltySlider.value);
    setStepPenalty(value);
    stepPenaltyValueSpan.textContent = value.toFixed(1);
});

canvas.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const gridX = Math.floor(mouseX / cellSize);
    const gridY = Math.floor(mouseY / cellSize);

    if (gridX >= 0 && gridX < gridSize && gridY >= 0 && gridY < gridSize) {
        const currentHover = { x: gridX, y: gridY };
        if (!hoveredCell || hoveredCell.x !== currentHover.x || hoveredCell.y !== currentHover.y) {
            hoveredCell = currentHover;
            // Only redraw if not currently animating to avoid visual conflicts
            if (!isAnimating) drawEverything();
        }
    } else {
        if (hoveredCell) {
            hoveredCell = null;
             // Only redraw if not currently animating
            if (!isAnimating) drawEverything();
        }
    }
});

canvas.addEventListener('mouseout', () => {
    if (hoveredCell) {
        hoveredCell = null;
        // Only redraw if not currently animating
        if (!isAnimating) drawEverything();
    }
});

// Listener for Cell Display Mode Select
cellDisplayModeSelect.addEventListener('change', () => {
    cellDisplayMode = cellDisplayModeSelect.value;
    console.log("Cell display mode changed to:", cellDisplayMode);

    // Redraw immediately to reflect the change
    // No need to check isLearning, drawEverything() should work regardless
    drawEverything();
});

// Event listener for Max Steps slider
maxStepsSlider.addEventListener('input', () => {
    const value = parseInt(maxStepsSlider.value, 10);
    maxStepsPerEpisode = value;
    maxStepsValueSpan.textContent = value.toString();
    // No reset needed here, will take effect on next episode end
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

// --- NEW: Reward Magnitude Slider Listeners ---
gemRewardSlider.addEventListener('input', () => {
    const value = parseFloat(gemRewardSlider.value);
    setGemRewardMagnitude(value);
    gemRewardValueSpan.textContent = value.toString();
    // Optional: Redraw values immediately if desired, though learning loop handles it too
    // if (!isLearning) drawEverything();
});

badStateRewardSlider.addEventListener('input', () => {
    const value = parseFloat(badStateRewardSlider.value);
    setBadStateRewardMagnitude(value);
    badStateRewardValueSpan.textContent = value.toString();
    // Optional: Redraw values immediately if desired
    // if (!isLearning) drawEverything();
});
// --- End Reward Magnitude Slider Listeners ---


// --- Initial Setup ---
async function initializeApp() {
    // NEW: Set initial theme based on localStorage or system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

    // Determine initial theme: saved preference > system preference > default (light)
    const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');
    setTheme(initialTheme); // Apply the initial theme (this also sets currentTheme var)

    stopButton.disabled = true;
    resetAgentButton.disabled = true;
    resetEnvironmentButton.disabled = true;

    gridSizeSlider.value = gridSize;
    stepPenaltySlider.value = parseFloat(stepPenaltySlider.value);
    maxStepsSlider.value = maxStepsPerEpisode;
    explorationStrategySelect.value = initialExplorationStrategy;
    algorithmSelect.value = initialAlgo;
    cellDisplayModeSelect.value = cellDisplayMode;
    terminateOnRewardCheckbox.checked = terminateOnGem;

    // Update the parameters in algorithms.js and environment.js to match initial UI/defaults
    updateLearningRate(initialLr);
    updateDiscountFactor(initialDf);
    updateExplorationRate(initialEr);
    updateSoftmaxBeta(initialBeta);
    updateExplorationStrategy(initialExplorationStrategy);
    updateSelectedAlgorithm(initialAlgo);
    setStepPenalty(parseFloat(stepPenaltySlider.value));
    setGemRewardMagnitude(initialGemReward);
    setBadStateRewardMagnitude(initialBadReward);
    updateTerminateOnGemSetting();
    maxStepsPerEpisode = parseInt(maxStepsSlider.value, 10);

    // Set initial UI values for new sliders
    gemRewardSlider.value = initialGemReward;
    gemRewardValueSpan.textContent = initialGemReward;
    badStateRewardSlider.value = initialBadReward;
    badStateRewardValueSpan.textContent = initialBadReward;

    // Set initial visibility for exploration AND specific LR controls based on INITIAL algorithm and STRATEGY
    const strategyField = explorationStrategySelect.parentElement;
    const epsilonField = epsilonSlider.parentElement.parentElement;

    // Hide all specific LR controls by default
    actorCriticLRControl.style.display = 'none';
    criticLRControl.style.display = 'none';
    srMLRControl.style.display = 'none';
    srWLRControl.style.display = 'none';
    // Hide exploration strategy/epsilon/softmax controls initially, show as needed
    if (strategyField) strategyField.style.display = 'none';
    if (epsilonField) epsilonField.style.display = 'none';
    softmaxBetaControl.style.display = 'none';
    // Hide main LR slider by default, show as needed
    if (lrControl) lrControl.style.display = 'none';


    // --- Set initial visibility for SR display options AND specific LR controls ---
    if (initialAlgo === 'sr') {
        srVectorAgentDisplayOption.style.display = '';
        srVectorHoverDisplayOption.style.display = '';
        // Show SR specific LR controls
        srMLRControl.style.display = '';
        srWLRControl.style.display = '';
        // Hide main LR control for SR
        if (lrControl) lrControl.style.display = 'none';
    } else {
        srVectorAgentDisplayOption.style.display = 'none';
        srVectorHoverDisplayOption.style.display = 'none';
         // Ensure the default selected value isn't one of the hidden SR options on load
         if (cellDisplayModeSelect.value === 'sr-vector' || cellDisplayModeSelect.value === 'sr-vector-hover') {
             cellDisplayModeSelect.value = 'values-color';
             cellDisplayMode = 'values-color';
         }
    }
    // --- End SR Display Option Visibility ---


    // --- Set initial visibility for Exploration AND AC/SR specific LR controls ---
    if (initialAlgo === 'actor-critic') {
         // AC uses softmax implicitly, show Beta control
        softmaxBetaControl.style.display = '';
        // Show Actor-Critic specific LR controls
        actorCriticLRControl.style.display = '';
        criticLRControl.style.display = '';
        // Hide main LR control for AC
        if (lrControl) lrControl.style.display = 'none';
        // Hide general exploration strategy/epsilon
        if (strategyField) strategyField.style.display = 'none';
        if (epsilonField) epsilonField.style.display = 'none';
    } else if (initialAlgo === 'sr') {
        // SR uses standard exploration, show strategy dropdown
        if (strategyField) strategyField.style.display = '';
         if (initialExplorationStrategy === 'epsilon-greedy') {
            if (epsilonField) epsilonField.style.display = '';
            softmaxBetaControl.style.display = 'none';
         } else { // Softmax
            if (epsilonField) epsilonField.style.display = 'none';
            softmaxBetaControl.style.display = '';
         }
         // SR specific LR controls (srWLRControl) are already shown above
         // Main LR control is already hidden above
    } else { // Handles QL, SARSA, ES, MC (non-AC, non-SR algos)
         // Show standard exploration strategy controls
         if (strategyField) strategyField.style.display = '';
         if (initialExplorationStrategy === 'epsilon-greedy') {
            if (epsilonField) epsilonField.style.display = '';
            softmaxBetaControl.style.display = 'none';
         } else { // Softmax
            if (epsilonField) epsilonField.style.display = 'none';
            softmaxBetaControl.style.display = '';
         }
         // Show main LR control for these algorithms
         if (lrControl) lrControl.style.display = '';
         // Ensure AC/SR specific LR controls remain hidden (done by default at start)
    }
    // --- End Initial Exploration/Specific LR Visibility ---

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