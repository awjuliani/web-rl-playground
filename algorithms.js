// Base Algorithm Class
class Algorithm {
    constructor(gridSize, config = {}) {
        this.gridSize = gridSize;
        this.config = {
            learningRate: 0.1,
            discountFactor: 0.9,
            explorationRate: 0.2,
            softmaxBeta: 1.0,
            explorationStrategy: 'epsilon-greedy',
            ...config
        };
        this.actions = ['up', 'down', 'left', 'right'];
        this.initializeTables(gridSize);
    }

    // Abstract methods to be implemented by subclasses
    initializeTables(gridSize) {
        throw new Error('initializeTables must be implemented by subclass');
    }

    chooseAction(state, takeActionFunc = null, agentPos = null) {
        throw new Error('chooseAction must be implemented by subclass');
    }

    learningStep(currentState, action, reward, nextState, done, takeActionFunc = null, agentPos = null) {
        throw new Error('learningStep must be implemented by subclass');
    }

    getBestActions(state, takeActionFunc = null, agentPos = null) {
        throw new Error('getBestActions must be implemented by subclass');
    }

    getValue(state) {
        throw new Error('getValue must be implemented by subclass');
    }

    // Common helper methods
    ensureStateInitialized(state) {
        // Default implementation - subclasses can override
    }

    calculateSoftmaxProbabilities(values, beta = null) {
        const numActions = values.length;
        if (numActions === 0) return {};

        const effectiveBeta = beta !== null ? beta : this.config.softmaxBeta;
        const maxVal = Math.max(...values);
        const expValues = values.map(v => Math.exp(effectiveBeta * (v - maxVal)));
        const sumExpValues = expValues.reduce((sum, val) => sum + val, 0);

        const probabilities = {};
        if (sumExpValues === 0 || !isFinite(sumExpValues)) {
            const uniformProb = 1.0 / numActions;
            this.actions.forEach(action => probabilities[action] = uniformProb);
        } else {
            const calculatedProbs = expValues.map(expVal => expVal / sumExpValues);
            this.actions.forEach((action, index) => {
                probabilities[action] = calculatedProbs[index];
            });
        }
        return probabilities;
    }

    chooseRandomAction() {
        return this.actions[Math.floor(Math.random() * this.actions.length)];
    }

    chooseGreedyAction(state, takeActionFunc = null, agentPos = null) {
        const bestActions = this.getBestActions(state, takeActionFunc, agentPos);
        return bestActions[Math.floor(Math.random() * bestActions.length)];
    }

    getActionProbabilities(state, takeActionFunc = null, agentPos = null) {
        const numActions = this.actions.length;
        if (numActions === 0) return {};

        let probabilities = {};
        const strategy = this.config.explorationStrategy;

        if (strategy === 'epsilon-greedy') {
            const bestActions = this.getBestActions(state, takeActionFunc, agentPos);
            const numBestActions = bestActions.length;
            const greedyProb = (1.0 - this.config.explorationRate);
            const exploreProb = this.config.explorationRate / numActions;

            this.actions.forEach(action => {
                if (bestActions.includes(action)) {
                    probabilities[action] = (greedyProb / numBestActions) + exploreProb;
                } else {
                    probabilities[action] = exploreProb;
                }
            });
        } else if (strategy === 'softmax') {
            const values = this.actions.map(action => this.getActionValue(state, action, takeActionFunc, agentPos));
            probabilities = this.calculateSoftmaxProbabilities(values);
        } else if (strategy === 'random') {
            const uniformProb = 1.0 / numActions;
            this.actions.forEach(action => probabilities[action] = uniformProb);
        } else if (strategy === 'greedy') {
            const bestActions = this.getBestActions(state, takeActionFunc, agentPos);
            const numBestActions = bestActions.length;
            const bestActionProb = 1.0 / numBestActions;
            this.actions.forEach(action => {
                probabilities[action] = bestActions.includes(action) ? bestActionProb : 0;
            });
        }

        return probabilities;
    }

    // Helper method to get action value - subclasses should override
    getActionValue(state, action, takeActionFunc = null, agentPos = null) {
        return 0;
    }

    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }

    reset() {
        // Default implementation - subclasses can override for episode-specific resets
    }
}

// Q-Learning Algorithm
class QLearningAlgorithm extends Algorithm {
    constructor(gridSize, config = {}) {
        super(gridSize, config);
    }

    initializeTables(gridSize) {
        this.qTable = {};
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                const state = `${x},${y}`;
                this.qTable[state] = {};
                for (const action of this.actions) {
                    this.qTable[state][action] = 0;
                }
            }
        }
    }

    ensureStateInitialized(state) {
        if (!this.qTable[state]) {
            this.qTable[state] = {};
            for (const action of this.actions) {
                this.qTable[state][action] = 0;
            }
        }
    }

    getActionValue(state, action) {
        this.ensureStateInitialized(state);
        return this.qTable[state][action];
    }

    getBestActions(state) {
        this.ensureStateInitialized(state);
        let maxValue = -Infinity;
        let bestActions = [];

        for (const action of this.actions) {
            const qValue = this.qTable[state][action];
            if (qValue > maxValue) {
                maxValue = qValue;
                bestActions = [action];
            } else if (qValue === maxValue) {
                bestActions.push(action);
            }
        }

        return bestActions.length > 0 ? bestActions : this.actions;
    }

    getValue(state) {
        this.ensureStateInitialized(state);
        let maxQ = -Infinity;
        for (const action of this.actions) {
            if (this.qTable[state][action] > maxQ) {
                maxQ = this.qTable[state][action];
            }
        }
        return maxQ === -Infinity ? 0 : maxQ;
    }

    chooseAction(state) {
        this.ensureStateInitialized(state);
        const strategy = this.config.explorationStrategy;

        if (strategy === 'epsilon-greedy') {
            if (Math.random() < this.config.explorationRate) {
                return this.chooseRandomAction();
            } else {
                return this.chooseGreedyAction(state);
            }
        } else if (strategy === 'softmax') {
            const qValues = this.actions.map(action => this.qTable[state][action]);
            const probabilities = this.calculateSoftmaxProbabilities(qValues);
            
            let cumulativeProb = 0;
            const randomSample = Math.random();
            for (let i = 0; i < this.actions.length; i++) {
                cumulativeProb += probabilities[this.actions[i]];
                if (randomSample < cumulativeProb) {
                    return this.actions[i];
                }
            }
            return this.actions[this.actions.length - 1];
        } else if (strategy === 'random') {
            return this.chooseRandomAction();
        } else if (strategy === 'greedy') {
            return this.chooseGreedyAction(state);
        }

        return this.chooseGreedyAction(state);
    }

    learningStep(currentState, action, reward, nextState, done) {
        this.ensureStateInitialized(currentState);
        this.ensureStateInitialized(nextState);

        const oldQValue = this.qTable[currentState][action];
        const bestNextActions = this.getBestActions(nextState);
        const maxNextQ = this.qTable[nextState][bestNextActions[0]];
        const tdTarget = reward + this.config.discountFactor * maxNextQ;
        const newQValue = oldQValue + this.config.learningRate * (tdTarget - oldQValue);
        
        this.qTable[currentState][action] = newQValue;
        return { needsStop: false };
    }
}

// SARSA Algorithm
class SarsaAlgorithm extends QLearningAlgorithm {
    learningStep(currentState, action, reward, nextState, done) {
        this.ensureStateInitialized(currentState);
        this.ensureStateInitialized(nextState);

        const oldQValue = this.qTable[currentState][action];
        const nextAction = this.chooseAction(nextState);
        const nextQ = this.qTable[nextState][nextAction];
        const tdTarget = reward + this.config.discountFactor * nextQ;
        const newQValue = oldQValue + this.config.learningRate * (tdTarget - oldQValue);
        
        this.qTable[currentState][action] = newQValue;
        return { needsStop: false };
    }
}

// Expected SARSA Algorithm
class ExpectedSarsaAlgorithm extends QLearningAlgorithm {
    learningStep(currentState, action, reward, nextState, done) {
        this.ensureStateInitialized(currentState);
        this.ensureStateInitialized(nextState);

        const oldQValue = this.qTable[currentState][action];
        const nextActionProbs = this.getActionProbabilities(nextState);
        let expectedNextQ = 0;
        
        for (const nextAction of this.actions) {
            const nextQValue = this.qTable[nextState][nextAction];
            const prob = nextActionProbs[nextAction];
            expectedNextQ += prob * nextQValue;
        }
        
        const tdTarget = reward + this.config.discountFactor * expectedNextQ;
        const newQValue = oldQValue + this.config.learningRate * (tdTarget - oldQValue);
        
        this.qTable[currentState][action] = newQValue;
        return { needsStop: false };
    }
}

// Monte Carlo Algorithm
class MonteCarloAlgorithm extends QLearningAlgorithm {
    constructor(gridSize, config = {}) {
        super(gridSize, config);
        this.currentEpisodeTrajectory = [];
    }

    learningStep(currentState, action, reward, nextState, done) {
        this.currentEpisodeTrajectory.push({ 
            state: currentState, 
            action: action, 
            reward: reward 
        });
        return { needsStop: false };
    }

    applyEpisodeUpdates() {
        let G = 0;
        for (let i = this.currentEpisodeTrajectory.length - 1; i >= 0; i--) {
            const { state, action, reward } = this.currentEpisodeTrajectory[i];
            G = reward + this.config.discountFactor * G;
            
            const oldQValue = this.qTable[state][action];
            const newQValue = oldQValue + this.config.learningRate * (G - oldQValue);
            this.qTable[state][action] = newQValue;
        }
        this.currentEpisodeTrajectory = [];
    }

    reset() {
        // Don't clear trajectory here - it's cleared in applyEpisodeUpdates
    }
}

// Actor-Critic Algorithm
class ActorCriticAlgorithm extends Algorithm {
    constructor(gridSize, config = {}) {
        super(gridSize, config);
    }

    initializeTables(gridSize) {
        this.vTable = {};
        this.hTable = {};
        
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                const state = `${x},${y}`;
                this.vTable[state] = 0;
                this.hTable[state] = {};
                for (const action of this.actions) {
                    this.hTable[state][action] = 0;
                }
            }
        }
    }

    ensureStateInitialized(state) {
        if (this.vTable[state] === undefined) {
            this.vTable[state] = 0;
        }
        if (!this.hTable[state]) {
            this.hTable[state] = {};
            for (const action of this.actions) {
                this.hTable[state][action] = 0;
            }
        }
    }

    getActionValue(state, action) {
        this.ensureStateInitialized(state);
        return this.hTable[state][action];
    }

    getBestActions(state) {
        this.ensureStateInitialized(state);
        let maxValue = -Infinity;
        let bestActions = [];

        for (const action of this.actions) {
            const preference = this.hTable[state][action];
            if (preference > maxValue) {
                maxValue = preference;
                bestActions = [action];
            } else if (preference === maxValue) {
                bestActions.push(action);
            }
        }

        return bestActions.length > 0 ? bestActions : this.actions;
    }

    getValue(state) {
        this.ensureStateInitialized(state);
        return this.vTable[state];
    }

    chooseAction(state) {
        this.ensureStateInitialized(state);
        const preferences = this.actions.map(action => this.hTable[state][action]);
        const probabilities = this.calculateSoftmaxProbabilities(preferences);
        
        let cumulativeProb = 0;
        const randomSample = Math.random();
        for (let i = 0; i < this.actions.length; i++) {
            cumulativeProb += probabilities[this.actions[i]];
            if (randomSample < cumulativeProb) {
                return this.actions[i];
            }
        }
        return this.actions[this.actions.length - 1];
    }

    getActionProbabilities(state) {
        this.ensureStateInitialized(state);
        const preferences = this.actions.map(action => this.hTable[state][action]);
        return this.calculateSoftmaxProbabilities(preferences);
    }

    learningStep(currentState, action, reward, nextState, done) {
        this.ensureStateInitialized(currentState);
        this.ensureStateInitialized(nextState);

        const V_current = this.vTable[currentState];
        const V_next = done ? 0 : this.vTable[nextState];
        const tdError = reward + this.config.discountFactor * V_next - V_current;

        // Critic Update
        this.vTable[currentState] = V_current + this.config.criticLearningRate * tdError;

        // Actor Update
        const preferences = this.actions.map(act => this.hTable[currentState][act]);
        const actionProbs = this.calculateSoftmaxProbabilities(preferences);

        for (const a of this.actions) {
            if (a === action) {
                this.hTable[currentState][a] += this.config.actorLearningRate * tdError * (1 - actionProbs[a]);
            } else {
                this.hTable[currentState][a] -= this.config.actorLearningRate * tdError * actionProbs[a];
            }
        }

        return { needsStop: false };
    }
}

// Successor Representation Algorithm
class SuccessorRepresentationAlgorithm extends Algorithm {
    constructor(gridSize, config = {}) {
        super(gridSize, config);
    }

    initializeTables(gridSize) {
        this.mTable = {};
        this.wTable = {};
        
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                const state = `${x},${y}`;
                this.wTable[state] = 0;
                this.mTable[state] = {};
                
                for (let x_prime = 0; x_prime < gridSize; x_prime++) {
                    for (let y_prime = 0; y_prime < gridSize; y_prime++) {
                        const state_prime = `${x_prime},${y_prime}`;
                        this.mTable[state][state_prime] = 0;
                    }
                }
            }
        }
    }

    ensureStateInitialized(state) {
        if (this.wTable[state] === undefined) {
            this.wTable[state] = 0;
        }
        if (!this.mTable[state]) {
            this.mTable[state] = {};
            for (let x_prime = 0; x_prime < this.gridSize; x_prime++) {
                for (let y_prime = 0; y_prime < this.gridSize; y_prime++) {
                    const state_prime = `${x_prime},${y_prime}`;
                    this.mTable[state][state_prime] = 0;
                }
            }
        }
    }

    calculateVValueSR(state) {
        let vValue = 0;
        if (!this.mTable[state]) return 0;

        for (let x_prime = 0; x_prime < this.gridSize; x_prime++) {
            for (let y_prime = 0; y_prime < this.gridSize; y_prime++) {
                const state_prime = `${x_prime},${y_prime}`;
                const m_s_sprime = this.mTable[state]?.[state_prime] ?? 0;
                const w_sprime = this.wTable[state_prime] ?? 0;
                vValue += m_s_sprime * w_sprime;
            }
        }
        return vValue;
    }

    calculateQValueSR(state, action, takeActionFunc, agentPos) {
        const { nextState: nextStateKey } = takeActionFunc(action, agentPos, this.gridSize);
        const V_next = this.calculateVValueSR(nextStateKey);
        const w_next = this.wTable[nextStateKey] ?? 0;
        return w_next + this.config.discountFactor * V_next;
    }

    getActionValue(state, action, takeActionFunc, agentPos) {
        return this.calculateQValueSR(state, action, takeActionFunc, agentPos);
    }

    getBestActions(state, takeActionFunc, agentPos) {
        this.ensureStateInitialized(state);
        let maxValue = -Infinity;
        let bestActions = [];

        for (const action of this.actions) {
            const qValue = this.calculateQValueSR(state, action, takeActionFunc, agentPos);
            if (qValue > maxValue) {
                maxValue = qValue;
                bestActions = [action];
            } else if (qValue === maxValue) {
                bestActions.push(action);
            }
        }

        return bestActions.length > 0 ? bestActions : this.actions;
    }

    getValue(state) {
        return this.calculateVValueSR(state);
    }

    chooseAction(state, takeActionFunc, agentPos) {
        this.ensureStateInitialized(state);
        const strategy = this.config.explorationStrategy;

        if (strategy === 'epsilon-greedy') {
            if (Math.random() < this.config.explorationRate) {
                return this.chooseRandomAction();
            } else {
                return this.chooseGreedyAction(state, takeActionFunc, agentPos);
            }
        } else if (strategy === 'softmax') {
            const qValues = this.actions.map(action => this.calculateQValueSR(state, action, takeActionFunc, agentPos));
            const probabilities = this.calculateSoftmaxProbabilities(qValues);
            
            let cumulativeProb = 0;
            const randomSample = Math.random();
            for (let i = 0; i < this.actions.length; i++) {
                cumulativeProb += probabilities[this.actions[i]];
                if (randomSample < cumulativeProb) {
                    return this.actions[i];
                }
            }
            return this.actions[this.actions.length - 1];
        } else if (strategy === 'random') {
            return this.chooseRandomAction();
        } else if (strategy === 'greedy') {
            return this.chooseGreedyAction(state, takeActionFunc, agentPos);
        }

        return this.chooseGreedyAction(state, takeActionFunc, agentPos);
    }

    learningStep(currentState, action, reward, nextState, done, takeActionFunc, agentPos) {
        this.ensureStateInitialized(currentState);
        this.ensureStateInitialized(nextState);

        // Update Reward Weights
        const current_w_next = this.wTable[nextState] ?? 0;
        const target_w = reward;
        const error_w = target_w - current_w_next;
        this.wTable[nextState] = current_w_next + this.config.srWWeightLearningRate * error_w;

        // Update Successor Representation
        if (this.mTable[currentState]) {
            for (let x_prime = 0; x_prime < this.gridSize; x_prime++) {
                for (let y_prime = 0; y_prime < this.gridSize; y_prime++) {
                    const state_prime = `${x_prime},${y_prime}`;
                    const M_s_prime = this.mTable[currentState]?.[state_prime] ?? 0;
                    const M_next_prime = done ? 0 : (this.mTable[nextState]?.[state_prime] ?? 0);
                    const indicator = (nextState === state_prime) ? 1 : 0;
                    const tdTarget_M = indicator + this.config.discountFactor * M_next_prime;
                    const tdError_M = tdTarget_M - M_s_prime;
                    this.mTable[currentState][state_prime] = M_s_prime + this.config.srMWeightLearningRate * tdError_M;
                }
            }
        }

        return { needsStop: false };
    }
}

// Algorithm Factory
class AlgorithmFactory {
    static create(algorithmType, gridSize, config = {}) {
        const algorithms = {
            'q-learning': QLearningAlgorithm,
            'sarsa': SarsaAlgorithm,
            'expected-sarsa': ExpectedSarsaAlgorithm,
            'monte-carlo': MonteCarloAlgorithm,
            'actor-critic': ActorCriticAlgorithm,
            'sr': SuccessorRepresentationAlgorithm
        };

        const AlgorithmClass = algorithms[algorithmType];
        if (!AlgorithmClass) {
            throw new Error(`Unknown algorithm type: ${algorithmType}`);
        }

        return new AlgorithmClass(gridSize, config);
    }
}

// Algorithm Manager to handle current algorithm and provide unified interface
class AlgorithmManager {
    constructor(initialAlgorithm, gridSize, config = {}) {
        this.currentAlgorithm = AlgorithmFactory.create(initialAlgorithm, gridSize, config);
        this.algorithmType = initialAlgorithm;
        this.gridSize = gridSize;
        this.baseConfig = config;
    }

    switchAlgorithm(newAlgorithmType, newConfig = {}) {
        const config = { ...this.baseConfig, ...newConfig };
        this.currentAlgorithm = AlgorithmFactory.create(newAlgorithmType, this.gridSize, config);
        this.algorithmType = newAlgorithmType;
    }

    updateGridSize(newGridSize) {
        this.gridSize = newGridSize;
        const config = this.currentAlgorithm.config;
        this.currentAlgorithm = AlgorithmFactory.create(this.algorithmType, newGridSize, config);
    }

    // Delegate methods to current algorithm
    chooseAction(state, takeActionFunc = null, agentPos = null) {
        return this.currentAlgorithm.chooseAction(state, takeActionFunc, agentPos);
    }

    learningStep(currentState, action, reward, nextState, done, takeActionFunc = null, agentPos = null) {
        return this.currentAlgorithm.learningStep(currentState, action, reward, nextState, done, takeActionFunc, agentPos);
    }

    getBestActions(state, takeActionFunc = null, agentPos = null) {
        return this.currentAlgorithm.getBestActions(state, takeActionFunc, agentPos);
    }

    getActionProbabilities(state, takeActionFunc = null, agentPos = null) {
        return this.currentAlgorithm.getActionProbabilities(state, takeActionFunc, agentPos);
    }

    getValue(state) {
        return this.currentAlgorithm.getValue(state);
    }

    updateConfig(newConfig) {
        this.baseConfig = { ...this.baseConfig, ...newConfig };
        this.currentAlgorithm.updateConfig(newConfig);
    }

    reset() {
        this.currentAlgorithm.reset();
    }

    // Special method for Monte Carlo
    applyEpisodeUpdates() {
        if (this.currentAlgorithm instanceof MonteCarloAlgorithm) {
            this.currentAlgorithm.applyEpisodeUpdates();
        }
    }

    // Getters for backward compatibility
    get qTable() { return this.currentAlgorithm.qTable || {}; }
    get vTable() { return this.currentAlgorithm.vTable || {}; }
    get hTable() { return this.currentAlgorithm.hTable || {}; }
    get mTable() { return this.currentAlgorithm.mTable || {}; }
    get wTable() { return this.currentAlgorithm.wTable || {}; }
}

// Export the main interface and factory
export const actions = ['up', 'down', 'left', 'right'];

// Create the algorithm manager instance
let algorithmManager = new AlgorithmManager('q-learning', 5, {
    learningRate: 0.1,
    actorLearningRate: 0.1,
    criticLearningRate: 0.1,
    srMWeightLearningRate: 0.1,
    srWWeightLearningRate: 0.1,
    discountFactor: 0.9,
    explorationRate: 0.2,
    softmaxBeta: 1.0,
    explorationStrategy: 'epsilon-greedy'
});

// Export the algorithm manager for direct access
export { algorithmManager };

// Replace all individual proxy objects with a factory function
function createTableProxy(tableGetter) {
    return new Proxy({}, {
        get: (target, prop) => {
            if (prop === Symbol.iterator || typeof prop === 'symbol') return undefined;
            return tableGetter()[prop];
        },
        has: (target, prop) => prop in tableGetter(),
        ownKeys: () => Object.keys(tableGetter()),
        getOwnPropertyDescriptor: (target, prop) => {
            if (prop in tableGetter()) {
                return { enumerable: true, configurable: true, value: tableGetter()[prop] };
            }
        }
    });
}

// Replace all the individual proxy exports with this:
export const qTable = createTableProxy(() => algorithmManager.qTable);
export const vTable = createTableProxy(() => algorithmManager.vTable);
export const hTable = createTableProxy(() => algorithmManager.hTable);
export const mTable = createTableProxy(() => algorithmManager.mTable);
export const wTable = createTableProxy(() => algorithmManager.wTable);

// Configuration getters
export const getConfig = () => algorithmManager.currentAlgorithm.config;
export const getLearningRate = () => algorithmManager.currentAlgorithm.config.learningRate;
export const getDiscountFactor = () => algorithmManager.currentAlgorithm.config.discountFactor;
export const getExplorationRate = () => algorithmManager.currentAlgorithm.config.explorationRate;
export const getSoftmaxBeta = () => algorithmManager.currentAlgorithm.config.softmaxBeta;
export const getExplorationStrategy = () => algorithmManager.currentAlgorithm.config.explorationStrategy;
export const getSelectedAlgorithm = () => algorithmManager.algorithmType;

// Specific learning rate getters for different algorithms
export const getActorLearningRate = () => algorithmManager.currentAlgorithm.config.actorLearningRate;
export const getCriticLearningRate = () => algorithmManager.currentAlgorithm.config.criticLearningRate;
export const getSRMWeightLearningRate = () => algorithmManager.currentAlgorithm.config.srMWeightLearningRate;
export const getSRWWeightLearningRate = () => algorithmManager.currentAlgorithm.config.srWWeightLearningRate;

// Backward compatibility - keep these exports for now until all references are updated
export let learningRate = 0.1;
export let actorLearningRate = 0.1;
export let criticLearningRate = 0.1;
export let srMWeightLearningRate = 0.1;
export let srWWeightLearningRate = 0.1;
export let discountFactor = 0.9;
export let explorationRate = 0.2;
export let softmaxBeta = 1.0;
export let explorationStrategy = 'epsilon-greedy';
export let selectedAlgorithm = 'q-learning';

// Main interface functions
export function initializeTables(gridSize) {
    algorithmManager.updateGridSize(gridSize);
}

export function learningStep(agentPos, gridSize, takeActionFunc, resetAgentFunc) {
    const currentState = `${agentPos.x},${agentPos.y}`;
    const action = algorithmManager.chooseAction(currentState, takeActionFunc, agentPos);
    const { nextState, reward, newAgentPos, done } = takeActionFunc(action, agentPos, gridSize);
    
    const result = algorithmManager.learningStep(currentState, action, reward, nextState, done, takeActionFunc, agentPos);
    
    return {
        ...result,
        newAgentPos: newAgentPos,
        reward: reward,
        done: done
    };
}

export function getBestActions(state, takeActionFunc = null, agentPos = null) {
    return algorithmManager.getBestActions(state, takeActionFunc, agentPos);
}

export function getActionProbabilities(state, takeActionFunc = null, agentPos = null) {
    return algorithmManager.getActionProbabilities(state, takeActionFunc, agentPos);
}

export function applyMonteCarloUpdates() {
    algorithmManager.applyEpisodeUpdates();
}

export function calculateVValueSR(state, currentMTable, currentWTable, gridSize) {
    if (algorithmManager.currentAlgorithm instanceof SuccessorRepresentationAlgorithm) {
        return algorithmManager.currentAlgorithm.calculateVValueSR(state);
    }
    return 0;
}

export function calculateQValueSR(state, action, currentMTable, currentWTable, gridSize, gamma, takeActionFunc, agentPos) {
    if (algorithmManager.currentAlgorithm instanceof SuccessorRepresentationAlgorithm) {
        return algorithmManager.currentAlgorithm.calculateQValueSR(state, action, takeActionFunc, agentPos);
    }
    return 0;
}

// Update functions
export function updateLearningRate(newLr) {
    learningRate = newLr;
    algorithmManager.updateConfig({ 
        learningRate: newLr,
        actorLearningRate: newLr,
        criticLearningRate: newLr,
        srMWeightLearningRate: newLr,
        srWWeightLearningRate: newLr
    });
}

export function updateActorLearningRate(newLr) {
    actorLearningRate = newLr;
    algorithmManager.updateConfig({ actorLearningRate: newLr });
}

export function updateCriticLearningRate(newLr) {
    criticLearningRate = newLr;
    algorithmManager.updateConfig({ criticLearningRate: newLr });
}

export function updateSRMLearningRate(newLr) {
    srMWeightLearningRate = newLr;
    algorithmManager.updateConfig({ srMWeightLearningRate: newLr });
}

export function updateSRWLearningRate(newLr) {
    srWWeightLearningRate = newLr;
    algorithmManager.updateConfig({ srWWeightLearningRate: newLr });
}

export function updateDiscountFactor(newDf) {
    discountFactor = newDf;
    algorithmManager.updateConfig({ discountFactor: newDf });
}

export function updateExplorationRate(newEr) {
    explorationRate = newEr;
    algorithmManager.updateConfig({ explorationRate: newEr });
}

export function updateSoftmaxBeta(newBeta) {
    softmaxBeta = newBeta;
    algorithmManager.updateConfig({ softmaxBeta: newBeta });
}

export function updateExplorationStrategy(newStrategy) {
    explorationStrategy = newStrategy;
    algorithmManager.updateConfig({ explorationStrategy: newStrategy });
}

export function updateSelectedAlgorithm(newAlgo) {
    selectedAlgorithm = newAlgo;
    algorithmManager.switchAlgorithm(newAlgo);
    console.log("Selected algorithm updated to:", selectedAlgorithm);
}

// Export classes for potential direct use
export { Algorithm, AlgorithmFactory, AlgorithmManager };

// Helper function to get value for a state - used by environment.js
export function getValueForState(state) {
    return algorithmManager.getValue(state);
} 