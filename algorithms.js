export let qTable = {}; // Q-table: state -> { action: qValue }
export let vTable = {}; // V-table (Critic): state -> value
export let hTable = {}; // Preference table (Actor): state -> { action: preference }
export let mTable = {}; // Successor Representation: state -> { nextState: discounted_occupancy }
export let wTable = {}; // Reward Weights: state -> weight (expected immediate reward)
export let learningRate = 0.1;
export let actorLearningRate = 0.1; // NEW: Actor specific LR
export let criticLearningRate = 0.1; // NEW: Critic specific LR
export let srMWeightLearningRate = 0.1; // NEW: SR Matrix specific LR
export let srWWeightLearningRate = 0.1; // RENAMED: SR Reward Weight specific LR (was R)
export let discountFactor = 0.9;
export let explorationRate = 0.2;
export let softmaxBeta = 1.0; // Temperature parameter for softmax
export let explorationStrategy = 'epsilon-greedy'; // 'epsilon-greedy' or 'softmax'
export let selectedAlgorithm = 'q-learning'; // Default algorithm
export let currentEpisodeTrajectory = []; // Stores (state, action, reward) for MC
let currentGridSizeForIteration = 0; // Store gridSize for SR iterations

export const actions = ['up', 'down', 'left', 'right'];

// --- Helper Functions ---
function ensureStateInitialized(state) {
    // Initialize Q-table entry if needed (for Q-learning, SARSA, etc.)
    if (!qTable[state]) {
        qTable[state] = {};
        for (const action of actions) {
            qTable[state][action] = 0;
        }
    }
    // Initialize V-table entry if needed (for Actor-Critic)
    if (vTable[state] === undefined) {
        vTable[state] = 0; // Initialize V(s) to 0
    }
    // Initialize H-table entry if needed (for Actor-Critic)
    if (!hTable[state]) {
        hTable[state] = {};
        for (const action of actions) {
            hTable[state][action] = 0; // Initialize preferences to 0
        }
    }
    // Initialize M-table entry if needed (for SR)
    if (!mTable[state]) {
        mTable[state] = {}; // M(state, :)
        // Initialize M(state, s_prime) = 0 for all s_prime in grid (done in initializeTables)
    }
    // Initialize W-table entry if needed (for SR)
    if (wTable[state] === undefined) {
        wTable[state] = 0; // Initialize w(s) to 0
    }
}


// --- Table Management --- Renamed from initializeQTable
export function initializeTables(gridSize) {
    qTable = {};
    vTable = {};
    hTable = {};
    mTable = {}; // NEW: Clear M-table
    wTable = {}; // NEW: Clear W-table
    currentEpisodeTrajectory = [];
    currentGridSizeForIteration = gridSize; // Store grid size

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const state = `${x},${y}`;
            ensureStateInitialized(state); // Initializes all tables for the state

            // Explicitly initialize M(s, s') = 0 for SR
            mTable[state] = {}; // M(state, :)
            for (let x_prime = 0; x_prime < gridSize; x_prime++) {
                for (let y_prime = 0; y_prime < gridSize; y_prime++) {
                    const state_prime = `${x_prime},${y_prime}`;
                    mTable[state][state_prime] = 0;
                }
            }
        }
    }
}

// --- SR Helper Functions ---
// Calculates V(s) using SR: V(s) = Sum_s' M(s, s') * w(s')
export function calculateVValueSR(state, currentMTable, currentWTable, gridSize) {
    let vValue = 0;
    if (!currentMTable[state]) return 0; // State not initialized in M

    for (let x_prime = 0; x_prime < gridSize; x_prime++) {
        for (let y_prime = 0; y_prime < gridSize; y_prime++) {
            const state_prime = `${x_prime},${y_prime}`;
            const m_s_sprime = currentMTable[state]?.[state_prime] ?? 0; // Default to 0 if undefined
            const w_sprime = currentWTable[state_prime] ?? 0; // Default to 0 if undefined
            vValue += m_s_sprime * w_sprime;
        }
    }
    return vValue;
}

// Calculates Q(s, a) using SR: Q(s, a) ~ w(s') + gamma * V(s') where s' is next state
// For this simple deterministic grid, reward comes upon *entering* s'.
// We use w(s') as the learned immediate reward estimate.
export function calculateQValueSR(state, action, currentMTable, currentWTable, gridSize, gamma, takeActionFuncForEnvLookup, currentAgentPosForLookup) {
    // Simulate the action to find the next state s' and immediate reward (for w learning)
    // Note: This uses the *actual* environment reward logic, which wTable tries to learn.
    // A pure SR approach might estimate Q differently, but this bridges SR with the env.
    const { nextState: nextStateKey, reward: immediateReward } = takeActionFuncForEnvLookup(action, currentAgentPosForLookup, gridSize);

    // Get the learned V-value of the next state using the SR components
    const V_next = calculateVValueSR(nextStateKey, currentMTable, currentWTable, gridSize);

    // Q(s,a) is estimated as the learned immediate reward of the next state + discounted value of next state
    // Alternatively: Q(s,a) = R(s,a,s') + gamma * V(s')
    const w_next = currentWTable[nextStateKey] ?? 0; // Use learned reward weight
    // return immediateReward + gamma * V_next; // Option 1: Use actual immediate reward
    return w_next + gamma * V_next; // Option 2: Use learned immediate reward (more SR-like)
}
// --- End SR Helper Functions ---


// --- Action Selection ---
// Returns an array of action(s) with the highest Q-value or preference
export function getBestActions(state, takeActionFuncForEnvLookup = null, currentAgentPosForLookup = null) { // Add optional args for SR Q calc
    ensureStateInitialized(state);

    let maxValue = -Infinity;
    let bestActions = [];

    if (selectedAlgorithm === 'actor-critic') {
        // Find best actions based on Actor's preferences (hTable)
        for (const action of actions) {
            const preference = hTable[state][action];
            if (preference > maxValue) {
                maxValue = preference;
                bestActions = [action];
            } else if (preference === maxValue) {
                bestActions.push(action);
            }
        }
    } else if (selectedAlgorithm === 'sr') {
        // SR: Find best actions based on Q-values computed from M and w
        for (const action of actions) {
            const qValue = calculateQValueSR(state, action, mTable, wTable, currentGridSizeForIteration, discountFactor, takeActionFuncForEnvLookup, currentAgentPosForLookup);
            if (qValue > maxValue) {
                maxValue = qValue;
                bestActions = [action];
            } else if (qValue === maxValue) {
                bestActions.push(action);
            }
        }
    } else {
        // Find best actions based on Q-values (qTable)
        for (const action of actions) {
            const qValue = qTable[state][action];
            if (qValue > maxValue) {
                maxValue = qValue;
                bestActions = [action];
            } else if (qValue === maxValue) {
                bestActions.push(action);
            }
        }
    }

    // If all values are the same (e.g., initial state), return all actions
    if (bestActions.length === actions.length && actions.length > 1) {
        // Optional: Could check if maxValue is actually 0 or the initial value
        // For simplicity, if all actions have the same max value, return all
    }
    // Ensure at least one action is returned if the state is initialized
    if (bestActions.length === 0 && actions.length > 0) {
        // This case should ideally not happen if ensureStateInitialized works,
        // but as a fallback, return a random action or all actions.
        return actions; // Or: [actions[Math.floor(Math.random() * actions.length)]]
    }


    return bestActions;
}

// --- Helper function for Softmax probability calculation (used by Softmax exploration and Actor-Critic) ---
function calculateSoftmaxProbabilities(values, beta = 1.0) {
    const numActions = values.length;
    if (numActions === 0) return {};

    // Find max for numerical stability
    const maxVal = Math.max(...values);
    // Calculate exponentiated values with beta, subtracting max
    const expValues = values.map(v => Math.exp(beta * (v - maxVal)));
    const sumExpValues = expValues.reduce((sum, val) => sum + val, 0);

    const probabilities = {};
    if (sumExpValues === 0 || !isFinite(sumExpValues)) {
        // Fallback to uniform if probabilities can't be computed
         const uniformProb = 1.0 / numActions;
         actions.forEach(action => probabilities[action] = uniformProb);
    } else {
        const calculatedProbs = expValues.map(expVal => expVal / sumExpValues);
         actions.forEach((action, index) => {
            probabilities[action] = calculatedProbs[index];
        });
        // Normalize probabilities just in case of floating point issues
        let sumProbs = Object.values(probabilities).reduce((sum, p) => sum + p, 0);
         if (Math.abs(sumProbs - 1.0) > 1e-6) {
            const factor = 1.0 / sumProbs;
            actions.forEach(action => { probabilities[action] *= factor; });
         }
    }
    return probabilities;
}


// Choose action needs to handle the array return if exploiting
export function chooseAction(state, takeActionFuncForEnvLookup = null, currentAgentPosForLookup = null) { // Add optional args for SR Q calc
    ensureStateInitialized(state);

    // Helper function for greedy action selection
    const chooseGreedyAction = () => {
        const bestActions = getBestActions(state, takeActionFuncForEnvLookup, currentAgentPosForLookup);
        return bestActions[Math.floor(Math.random() * bestActions.length)];
    };

    // Helper function for random action selection
    const chooseRandomAction = () => {
        return actions[Math.floor(Math.random() * actions.length)];
    };

    if (selectedAlgorithm === 'actor-critic') {
        // Actor-Critic always uses its learned policy (softmax over preferences)
        const preferences = actions.map(action => hTable[state][action]);
        const probabilities = calculateSoftmaxProbabilities(preferences, softmaxBeta);
        // ... (sampling logic using probabilities) ...
        let cumulativeProb = 0;
        const randomSample = Math.random();
        for (let i = 0; i < actions.length; i++) {
            cumulativeProb += probabilities[actions[i]];
            if (randomSample < cumulativeProb) {
                return actions[i];
            }
        }
        // Fallback in case of numerical issues
        return actions[actions.length - 1];

    } else if (selectedAlgorithm === 'sr') {
        // SR: Choose action based on computed Q-values and exploration strategy
        const getQ = (action) => calculateQValueSR(state, action, mTable, wTable, currentGridSizeForIteration, discountFactor, takeActionFuncForEnvLookup, currentAgentPosForLookup);

        if (explorationStrategy === 'epsilon-greedy') {
            if (Math.random() < explorationRate) {
                return chooseRandomAction();
            } else {
                return chooseGreedyAction(); // Use helper
            }
        } else if (explorationStrategy === 'softmax') {
            const qValues = actions.map(action => getQ(action));
            const probabilities = calculateSoftmaxProbabilities(qValues, softmaxBeta);
            // ... (sampling logic using probabilities) ...
            let cumulativeProb = 0;
            const randomSample = Math.random();
            for (let i = 0; i < actions.length; i++) {
                cumulativeProb += probabilities[actions[i]];
                if (randomSample < cumulativeProb) {
                    return actions[i];
                }
            }
            // Fallback
            return actions[actions.length - 1];
        } else if (explorationStrategy === 'random') { // NEW: Random Strategy for SR
             return chooseRandomAction();
        } else if (explorationStrategy === 'greedy') { // NEW: Greedy Strategy for SR
             return chooseGreedyAction();
        } else {
            // Fallback SR exploration
            console.error("Unknown exploration strategy for SR:", explorationStrategy);
            return chooseGreedyAction(); // Fallback to greedy
        }

    } else { // Handles Q-Learning, SARSA, Expected SARSA, Monte Carlo
        if (explorationStrategy === 'epsilon-greedy') {
             if (Math.random() < explorationRate) {
                return chooseRandomAction();
             } else {
                return chooseGreedyAction(); // Use helper
             }
        } else if (explorationStrategy === 'softmax') {
            const qValues = actions.map(action => qTable[state][action]);
            const probabilities = calculateSoftmaxProbabilities(qValues, softmaxBeta);
            // ... (sampling logic using probabilities) ...
            let cumulativeProb = 0;
            const randomSample = Math.random();
            for (let i = 0; i < actions.length; i++) {
                cumulativeProb += probabilities[actions[i]];
                if (randomSample < cumulativeProb) {
                    return actions[i];
                }
            }
            // Fallback
            return actions[actions.length - 1];
        } else if (explorationStrategy === 'random') { // NEW: Random Strategy for TD/MC
             return chooseRandomAction();
        } else if (explorationStrategy === 'greedy') { // NEW: Greedy Strategy for TD/MC
             return chooseGreedyAction();
        } else {
            // Fallback if exploration strategy is unknown for MC/TD
            console.error("Unknown exploration strategy for MC/TD:", explorationStrategy);
             return chooseGreedyAction(); // Fallback to greedy
        }
    }
}

// --- Get Action Probabilities ---
export function getActionProbabilities(state, takeActionFuncForEnvLookup = null, currentAgentPosForLookup = null) { // Add optional args for SR Q calc
    ensureStateInitialized(state);
    const numActions = actions.length;
    if (numActions === 0) return {};

    let probabilities = {};

    if (selectedAlgorithm === 'actor-critic') {
        // Actor-Critic policy probabilities from hTable preferences
        const preferences = actions.map(action => hTable[state][action]);
        probabilities = calculateSoftmaxProbabilities(preferences, softmaxBeta);

    } else if (selectedAlgorithm === 'sr') {
        // SR: Calculate probabilities based on computed Q-values and exploration strategy
        const getQ = (action) => calculateQValueSR(state, action, mTable, wTable, currentGridSizeForIteration, discountFactor, takeActionFuncForEnvLookup, currentAgentPosForLookup);

        if (explorationStrategy === 'epsilon-greedy') {
            const bestActions = getBestActions(state, takeActionFuncForEnvLookup, currentAgentPosForLookup);
            const numBestActions = bestActions.length;
            const greedyProb = (1.0 - explorationRate);
            const exploreProb = explorationRate / numActions;

            actions.forEach(action => {
                if (bestActions.includes(action)) {
                    probabilities[action] = (greedyProb / numBestActions) + exploreProb;
                } else {
                    probabilities[action] = exploreProb;
                }
            });
        } else if (explorationStrategy === 'softmax') {
            const qValues = actions.map(action => getQ(action));
            probabilities = calculateSoftmaxProbabilities(qValues, softmaxBeta);
        } else if (explorationStrategy === 'random') { // NEW: Random Strategy for SR
            const uniformProb = 1.0 / numActions;
            actions.forEach(action => probabilities[action] = uniformProb);
        } else if (explorationStrategy === 'greedy') { // NEW: Greedy Strategy for SR
            const bestActions = getBestActions(state, takeActionFuncForEnvLookup, currentAgentPosForLookup);
            const numBestActions = bestActions.length;
            const bestActionProb = 1.0 / numBestActions;
            actions.forEach(action => {
                probabilities[action] = bestActions.includes(action) ? bestActionProb : 0;
            });
        } else {
            // Fallback SR probabilities
             console.error("Unknown exploration strategy for SR probabilities:", explorationStrategy);
             const uniformProb = 1.0 / numActions;
             actions.forEach(action => probabilities[action] = uniformProb);
        }

    } else { // Handles Q-Learning, SARSA, Expected SARSA, Monte Carlo
         if (explorationStrategy === 'epsilon-greedy') {
            const bestActions = getBestActions(state);
            const numBestActions = bestActions.length;
            const greedyProb = (1.0 - explorationRate);
            const exploreProb = explorationRate / numActions;

            actions.forEach(action => {
                if (bestActions.includes(action)) {
                    probabilities[action] = (greedyProb / numBestActions) + exploreProb;
                } else {
                    probabilities[action] = exploreProb;
                }
            });
         } else if (explorationStrategy === 'softmax') {
            const qValues = actions.map(action => qTable[state][action]);
            probabilities = calculateSoftmaxProbabilities(qValues, softmaxBeta);
         } else if (explorationStrategy === 'random') { // NEW: Random Strategy for TD/MC
            const uniformProb = 1.0 / numActions;
            actions.forEach(action => probabilities[action] = uniformProb);
         } else if (explorationStrategy === 'greedy') { // NEW: Greedy Strategy for TD/MC
            const bestActions = getBestActions(state);
            const numBestActions = bestActions.length;
            const bestActionProb = 1.0 / numBestActions;
            actions.forEach(action => {
                probabilities[action] = bestActions.includes(action) ? bestActionProb : 0;
            });
         } else {
             // Fallback for unknown strategy with MC/TD
             console.error("Unknown exploration strategy for MC/TD probabilities:", explorationStrategy);
             const uniformProb = 1.0 / numActions;
             actions.forEach(action => probabilities[action] = uniformProb);
         }
    }

    // Final check for probability sum (can help catch issues)
    let sumProbs = Object.values(probabilities).reduce((sum, p) => sum + p, 0);
    if (Math.abs(sumProbs - 1.0) > 1e-6 && numActions > 0) {
        console.warn(`Probabilities do not sum to 1 for state ${state}, strategy ${explorationStrategy}. Sum: ${sumProbs}. Normalizing.`);
        // Simple normalization as a fallback
        const factor = 1.0 / sumProbs;
        actions.forEach(action => { probabilities[action] *= factor; });
    }

    return probabilities;
}

// --- Learning Step ---
export function learningStep(agentPos, gridSize, takeActionFunc, resetAgentFunc) {
    const currentState = `${agentPos.x},${agentPos.y}`;
    ensureStateInitialized(currentState);

    // Pass necessary functions/state for SR Q-value calculation during action selection
    const action = chooseAction(currentState, takeActionFunc, agentPos);
    const { nextState, reward, newAgentPos, done } = takeActionFunc(action, agentPos, gridSize);

    ensureStateInitialized(nextState); // Ensure next state is initialized before using it

    // --- Algorithm-Specific Updates ---
    if (selectedAlgorithm === 'monte-carlo') {
        // Store step for Monte Carlo (updates happen later)
        currentEpisodeTrajectory.push({ state: currentState, action: action, reward: reward });
    } else if (selectedAlgorithm === 'actor-critic') {
        // --- Actor-Critic Update ---
        const V_current = vTable[currentState];
        // If done, the value of the terminal state is 0
        const V_next = done ? 0 : vTable[nextState];

        // 1. Calculate TD Error (Advantage in this simple case)
        const tdError = reward + discountFactor * V_next - V_current;

        // 2. Critic Update (Update V(s)) using criticLearningRate
        vTable[currentState] = V_current + criticLearningRate * tdError; // MODIFIED

        // 3. Actor Update (Update preferences H(s,a)) using actorLearningRate
        // Get action probabilities based on current preferences
        const preferences = actions.map(act => hTable[currentState][act]);
        const actionProbs = calculateSoftmaxProbabilities(preferences, softmaxBeta);

        for (const a of actions) {
            if (a === action) {
                // Increase preference for the action taken proportionally to TD error and (1 - prob)
                hTable[currentState][a] += actorLearningRate * tdError * (1 - actionProbs[a]); // MODIFIED
            } else {
                 // Decrease preference for other actions proportionally to TD error and prob
                 hTable[currentState][a] -= actorLearningRate * tdError * actionProbs[a]; // MODIFIED
            }
        }
        // --- End Actor-Critic Update ---

    } else if (selectedAlgorithm === 'sr') {
        // --- Successor Representation Update ---
        // 1. Update Reward Weights w(s') using srWWeightLearningRate
        const current_w_next = wTable[nextState] ?? 0;
        const target_w = reward; // Target for w(s') is the immediate reward received
        const error_w = target_w - current_w_next;
        wTable[nextState] = current_w_next + srWWeightLearningRate * error_w; // MODIFIED (Variable Renamed)

        // 2. Update Successor Representation M(s, s_prime) using srMWeightLearningRate
        //    M(s, s_prime) <-- M(s, s_prime) + alpha_M * [ Indicator(nextState == s_prime) + gamma * M(nextState, s_prime) - M(s, s_prime) ]
        if (mTable[currentState]) { // Check if current state exists in M
            for (let x_prime = 0; x_prime < gridSize; x_prime++) {
                for (let y_prime = 0; y_prime < gridSize; y_prime++) {
                    const state_prime = `${x_prime},${y_prime}`;

                    // Ensure M entries exist, default to 0 if not
                    const M_s_prime = mTable[currentState]?.[state_prime] ?? 0;
                    const M_next_prime = done ? 0 : (mTable[nextState]?.[state_prime] ?? 0); // M=0 if next state is terminal

                    const indicator = (nextState === state_prime) ? 1 : 0;

                    const tdTarget_M = indicator + discountFactor * M_next_prime;
                    const tdError_M = tdTarget_M - M_s_prime;

                    mTable[currentState][state_prime] = M_s_prime + srMWeightLearningRate * tdError_M; // (No change here)
                }
            }
        } else {
            console.warn(`SR: State ${currentState} not found in mTable during update.`);
        }
        // --- End SR Update ---

    } else {
        // --- TD Learning Updates (Q-Learning, SARSA, Expected SARSA) ---
        const oldQValue = qTable[currentState][action];
        let tdTarget;

        // Calculate TD Target based on the specific TD algorithm
        if (selectedAlgorithm === 'q-learning') {
            const bestNextActions = getBestActions(nextState);
            // Ensure next state Q values are initialized before accessing
            ensureStateInitialized(nextState); // Added safety check
            const maxNextQ = qTable[nextState][bestNextActions[0]]; // Assumes getBestActions returns at least one
            tdTarget = reward + discountFactor * maxNextQ;
        } else if (selectedAlgorithm === 'sarsa') {
            // Ensure next state Q values are initialized before choosing next action based on them
            ensureStateInitialized(nextState); // Added safety check
            const nextAction = chooseAction(nextState); // chooseAction selects based on qTable/policy
            const nextQ = qTable[nextState][nextAction];
            tdTarget = reward + discountFactor * nextQ;
        } else if (selectedAlgorithm === 'expected-sarsa') {
             // Ensure next state Q values are initialized before calculating expected value
            ensureStateInitialized(nextState); // Added safety check
            const nextActionProbs = getActionProbabilities(nextState);
            let expectedNextQ = 0;
            for (const nextAction of actions) {
                const nextQValue = qTable[nextState][nextAction];
                const prob = nextActionProbs[nextAction];
                expectedNextQ += prob * nextQValue;
            }
            tdTarget = reward + discountFactor * expectedNextQ;
        } else {
            console.error("Unknown TD algorithm selected for update:", selectedAlgorithm);
            return { needsStop: true, newAgentPos: agentPos, done: false };
        }

        const newQValue = oldQValue + learningRate * (tdTarget - oldQValue); // Uses general learningRate
        qTable[currentState][action] = newQValue;
        // --- End TD Learning Updates ---
    }

    return { needsStop: false, newAgentPos: newAgentPos, reward: reward, done: done };
}

// --- Monte Carlo Update --- Apply updates after an episode finishes
export function applyMonteCarloUpdates() {
    let G = 0;
    // const visited = new Set(); // Uncomment for First-Visit MC

    for (let i = currentEpisodeTrajectory.length - 1; i >= 0; i--) {
        const { state, action, reward } = currentEpisodeTrajectory[i];
        // const stateActionPair = `${state}|${action}`; // For First-Visit MC

        G = reward + discountFactor * G;

        // Every-Visit MC (current implementation):
        // if (!visited.has(stateActionPair)) { // Uncomment block for First-Visit MC
        //    visited.add(stateActionPair);
             const oldQValue = qTable[state][action];
             const newQValue = oldQValue + learningRate * (G - oldQValue);
             qTable[state][action] = newQValue;
        // } // End First-Visit MC block
    }
    currentEpisodeTrajectory = [];
}

// Update Parameters
export function updateLearningRate(newLr) {
    learningRate = newLr;
    // ALSO update the specific learning rates to this new baseline
    // The specific sliders can then override these if the user adjusts them
    actorLearningRate = newLr;
    criticLearningRate = newLr;
    srMWeightLearningRate = newLr;
    srWWeightLearningRate = newLr; // MODIFIED (Variable Renamed)
}

// NEW: Update functions for specific learning rates
export function updateActorLearningRate(newLr) {
    actorLearningRate = newLr;
}
export function updateCriticLearningRate(newLr) {
    criticLearningRate = newLr;
}
export function updateSRMLearningRate(newLr) {
    srMWeightLearningRate = newLr;
}
export function updateSRWLearningRate(newLr) { // RENAMED (was updateSRRLearningRate)
    srWWeightLearningRate = newLr; // MODIFIED (Variable Renamed)
}

export function updateDiscountFactor(newDf) {
    discountFactor = newDf;
}

export function updateExplorationRate(newEr) {
    explorationRate = newEr;
}

export function updateSoftmaxBeta(newBeta) {
    softmaxBeta = newBeta;
}

export function updateExplorationStrategy(newStrategy) {
    explorationStrategy = newStrategy;
}

export function updateSelectedAlgorithm(newAlgo) {
    selectedAlgorithm = newAlgo;
    console.log("Selected algorithm updated to:", selectedAlgorithm);
} 