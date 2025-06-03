// --- Environment Setup ---
export const canvas = document.getElementById('gridCanvas');
export const ctx = canvas.getContext('2d');
export const gridColor = '#ccc';

// Reward state constants
export const REWARD_EMPTY = 0;
export const REWARD_GEM = 1;
export const REWARD_BAD = -1;
export const REWARD_WALL = 2;

// Reward magnitudes for terminal states (NOW VARIABLES)
// export const REWARD_MAGNITUDE_GEM = 10; // OLD CONSTANT
// export const REWARD_MAGNITUDE_BAD = -10; // OLD CONSTANT
export let rewardMagnitudeGem = 10;
export let rewardMagnitudeBad = -10;

// --- SVGs ---
const agentSvgString = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="70">🤖</text>
</svg>`;

const rewardSvgString = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
   <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="70">💎</text>
</svg>`;

const badStateSvgString = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="70">☠️</text>
</svg>`;

const wallSvgString = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="70">🚧</text>
</svg>`;

export let agentImage = new Image();
export let rewardImage = new Image();
export let badStateImage = new Image(); // Image for the bad state
export let wallImage = new Image(); // NEW: Image for the wall state
let agentLoaded = false;
let rewardLoaded = false;
let badStateLoaded = false; // Flag for bad state image
let wallLoaded = false; // NEW: Flag for wall image
let stepPenalty = -0.1; // Default step penalty

// --- State ---
export let agentPos = { x: 0, y: 0 };
export let startPos = { x: 0, y: 0 }; // NEW: Track start position
// Initial reward position depends on gridSize, set during initialization/reset
export let gridRewards = []; // 2D array to store reward state for each cell (0: empty, 1: gem, -1: bad)
let terminateOnGem = true; // Setting to control termination

// --- Setter for Agent Position ---
export function setAgentPos(newPos) {
    if (typeof newPos?.x === 'number' && typeof newPos?.y === 'number') {
        agentPos = { ...newPos }; // Update the environment's agent position
    } else {
        console.error("Invalid newPos provided to setAgentPos:", newPos);
    }
}

// --- Setter for Start Position --- // NEW
export function setStartPos(newPos, gridSize) {
    if (typeof newPos?.x === 'number' && typeof newPos?.y === 'number' &&
        newPos.x >= 0 && newPos.x < gridSize &&
        newPos.y >= 0 && newPos.y < gridSize) {
        // Check if the target cell is empty (not a gem, bad state, or wall)
        if (gridRewards[newPos.y][newPos.x] === REWARD_EMPTY) {
            startPos = { ...newPos }; // Update the environment's start position
            console.log("Start position set to:", startPos);
            return true; // Indicate success
        } else {
            console.warn("Cannot set start position on a non-empty cell (gem, bad, or wall)."); // MODIFIED Warning
            return false; // Indicate failure
        }
    } else {
        console.error("Invalid newPos provided to setStartPos:", newPos);
        return false; // Indicate failure
    }
}

// --- Setter for Terminate Setting ---
export function setTerminateOnGem(value) {
    terminateOnGem = !!value; // Ensure boolean
}

// --- Setter for Step Penalty ---
export function setStepPenalty(value) {
    const newPenalty = parseFloat(value);
    if (!isNaN(newPenalty)) {
        stepPenalty = newPenalty;
        console.log("Step penalty updated to:", stepPenalty);
    } else {
        console.warn("Invalid step penalty value:", value);
    }
}

// --- NEW: Setter for Gem Reward Magnitude ---
export function setGemRewardMagnitude(value) {
    const newReward = parseFloat(value);
    if (!isNaN(newReward) && newReward >= 0) { // Ensure non-negative
        rewardMagnitudeGem = newReward;
        console.log("Gem reward magnitude updated to:", rewardMagnitudeGem);
    } else {
        console.warn("Invalid gem reward magnitude value:", value);
    }
}

// --- NEW: Setter for Bad State Reward Magnitude ---
export function setBadStateRewardMagnitude(value) {
    const newReward = parseFloat(value);
    if (!isNaN(newReward) && newReward <= 0) { // Ensure non-positive
        rewardMagnitudeBad = newReward;
        console.log("Bad state reward magnitude updated to:", rewardMagnitudeBad);
    } else {
        console.warn("Invalid bad state reward magnitude value:", value);
    }
}

// Helper to initialize or resize gridRewards
export function initializeGridRewards(size) {
    gridRewards = Array(size).fill(null).map(() => Array(size).fill(REWARD_EMPTY));
    // Set default gem and bad state locations if needed, or leave empty
    // Example: Place a gem in the bottom right and a bad state in the center
    if (size > 0) {
        // Ensure default placements don't overlap if size is small
        const gemX = size - 1;
        const gemY = size - 1;
        const badX = Math.floor(size / 2);
        const badY = Math.floor(size / 2);

        if (gemX === badX && gemY === badY && size > 1) {
            // If they overlap, maybe move the bad state
            gridRewards[badY - 1][badX] = REWARD_BAD;
        } else if (size > 2) {
             gridRewards[badY][badX] = REWARD_BAD;
        }
        gridRewards[gemY][gemX] = REWARD_GEM;

        // Example wall placement (optional, you can remove or change this)
        if (size > 3) {
            gridRewards[1][2] = REWARD_WALL;
            gridRewards[3][2] = REWARD_WALL;
        }
    }
    console.log("Initialized gridRewards for size:", size); // Add log
}

// --- Grid Management ---
export function updateGridSize(gridSizeInput, currentGridSize) {
    const newSize = parseInt(gridSizeInput.value, 10);
    if (!isNaN(newSize) && newSize >= 2 && newSize <= 20) {
        const newCellSize = canvas.width / newSize;
        // initializeGridRewards(newSize); // REMOVE: Initialization will happen in reset or slider handler
        console.log("Grid size updated to:", newSize);
        return { gridSize: newSize, cellSize: newCellSize, updated: true };
    } else {
        console.warn("Invalid grid size input:", gridSizeInput.value);
        // Use currentGridSize if it's a slider interaction, otherwise maybe default?
        // For slider, setting value back might be confusing. Let's rely on the main 'gridSize' var.
        // gridSizeInput.value = currentGridSize; // Don't reset slider input value directly
        return { gridSize: currentGridSize, cellSize: canvas.width / currentGridSize, updated: false };
    }
}

export function resetAgent() {
    // agentPos = { x: 0, y: 0 }; // OLD
    agentPos = { ...startPos }; // MODIFIED: Reset to current startPos
    // REMOVED conditional initialization logic
}

// --- Drawing Functions ---
export function drawGrid(gridSize, cellSize, hoveredCell) {
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;

    // Draw grid lines first
    for (let i = 0; i <= gridSize; i++) {
        ctx.beginPath();
        ctx.moveTo(i * cellSize, 0);
        ctx.lineTo(i * cellSize, canvas.height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * cellSize);
        ctx.lineTo(canvas.width, i * cellSize);
        ctx.stroke();
    }

    // --- Draw Start Location Indicator ---
    if (gridSize > 0 && cellSize > 0) {
        // const startX = 0; // OLD
        // const startY = 0; // OLD
        const startX = startPos.x; // MODIFIED: Use startPos
        const startY = startPos.y; // MODIFIED: Use startPos
        const emojiSize = cellSize * 0.6;
        const emojiX = startX * cellSize + (cellSize / 2);
        const emojiY = startY * cellSize + (cellSize / 2);

        ctx.save();
        ctx.font = `${emojiSize}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = 0.5;
        ctx.fillText('🏠', emojiX, emojiY);
        ctx.restore();
    }
    // --- End Start Location Indicator ---

    // Draw highlight on hovered cell (draw this *after* grid lines but *before* other items?)
    // Let's draw it here, so it's under the agent/items but over the base grid/values/policy
    if (hoveredCell) {
        ctx.fillStyle = 'rgba(255, 255, 0, 0.3)'; // Semi-transparent yellow
        ctx.fillRect(hoveredCell.x * cellSize, hoveredCell.y * cellSize, cellSize, cellSize);
    }
}

export function drawAgent(pos, cellSize, visualPos = null) {
    if (agentLoaded) {
        const drawX = visualPos ? visualPos.x : pos.x;
        const drawY = visualPos ? visualPos.y : pos.y;
        ctx.drawImage(agentImage, drawX * cellSize, drawY * cellSize, cellSize, cellSize);
    }
}

// Draw gems, bad states, and walls based on gridRewards
export function drawCellStates(gridSize, cellSize) {
    // Ensure all necessary images are loaded
    if (!rewardLoaded || !badStateLoaded || !wallLoaded) return;

    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const state = gridRewards[y][x];
            if (state === REWARD_GEM) {
                ctx.drawImage(rewardImage, x * cellSize, y * cellSize, cellSize, cellSize);
            } else if (state === REWARD_BAD) {
                ctx.drawImage(badStateImage, x * cellSize, y * cellSize, cellSize, cellSize);
            } else if (state === REWARD_WALL) { // NEW: Draw wall
                ctx.drawImage(wallImage, x * cellSize, y * cellSize, cellSize, cellSize);
            }
            // REWARD_EMPTY (0) draws nothing
        }
    }
}

// --- Value Function Calculation & Drawing (Moved from algorithms.js) ---
import {
    actions, getBestActions, getActionProbabilities, qTable, vTable, hTable,
    mTable, wTable, // Import SR tables
    getSelectedAlgorithm, // Use getter instead of selectedAlgorithm
    calculateVValueSR, // Import SR V calculation helper
    getDiscountFactor, // Use getter instead of discountFactor
    getValueForState // Import the helper function
} from './algorithms.js';

// Helper function to get the Value (max Q or V(s)) for a state
function getValue(state, qTable, vTable, mTable, wTable, currentAlgorithm, gridSize) {
    try {
        // Use the helper function
        return getValueForState(state);
    } catch (error) {
        console.warn('Error accessing algorithm value, falling back to direct table access:', error);
        
        // Fallback to direct table access
        if (currentAlgorithm === 'actor-critic') {
            return vTable[state] !== undefined ? vTable[state] : 0;
        } else if (currentAlgorithm === 'sr') {
            return calculateVValueSR(state, mTable, wTable, gridSize, getDiscountFactor()); // Use getter
        } else {
            if (!qTable || !qTable[state]) return 0;
            let maxQ = -Infinity;
            for (const action of actions) {
                if (qTable[state][action] !== undefined && qTable[state][action] > maxQ) {
                    maxQ = qTable[state][action];
                }
            }
            return maxQ === -Infinity ? 0 : maxQ;
        }
    }
}

// Helper function to find min/max V(s) across the grid
function getMinMaxValues(gridSize, qTable, vTable, mTable, wTable, currentAlgorithm) {
    let minV = Infinity;
    let maxV = -Infinity;
    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const state = `${x},${y}`;
            // Pass all potentially needed tables and the algorithm to getValue
            const value = getValue(state, qTable, vTable, mTable, wTable, currentAlgorithm, gridSize);
            if (value < minV) minV = value;
            if (value > maxV) maxV = value;
        }
    }
    // Handle cases where min/max weren't updated (e.g., all zeros)
    if (minV === Infinity || maxV === -Infinity) {
         minV = Math.min(0, maxV === -Infinity ? 0 : maxV); // If maxV is still -inf, minV should be 0
         maxV = Math.max(0, minV === Infinity ? 0 : minV); // If minV is still inf, maxV should be 0
     }
     // Ensure min/max range isn't just zero
     if (Math.abs(maxV - minV) < 1e-6) {
         if (Math.abs(maxV) < 1e-6) { // Both are zero
             minV = -0.1; // Give a small range around zero
             maxV = 0.1;
         } else if (maxV > 0) { // Both are positive
             minV = 0;
         } else { // Both are negative
             maxV = 0;
         }
     }
    return { minV, maxV };
}

// Helper function to map a value to a Red-Blue-White color using CSS variables
function valueToColor(value, minV, maxV) {
    const epsilon = 1e-6; // Small tolerance for zero check

    // If range is zero or value is close to zero, use the zero-value background
    if (Math.abs(maxV - minV) < epsilon || Math.abs(value) < epsilon) {
        return getComputedStyle(document.documentElement).getPropertyValue('--color-value-zero-bg');
    }

    let colorVar;

    if (value > 0) {
        colorVar = '--color-value-pos-bg';
        // Ensure maxV is positive and significantly different from 0
        if (maxV <= epsilon) return getComputedStyle(document.documentElement).getPropertyValue('--color-value-zero-bg');
        // Normalize intensity based on positive range [0, maxV]
        const intensity = Math.min(1, Math.max(0, value / maxV));
        // We need to adjust alpha based on intensity
        // Get the base color string (e.g., "rgba(r, g, b, alpha)")
        const baseColor = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();
        // Extract existing alpha and replace it
        const rgbaMatch = baseColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (rgbaMatch) {
            // Adjust alpha based on intensity (e.g., scale original alpha)
            const originalAlpha = parseFloat(rgbaMatch[4] || '1');
            const newAlpha = Math.max(0.1, originalAlpha * intensity); // Ensure minimum visibility
            return `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${newAlpha})`;
        }
        return baseColor; // Fallback to original color

    } else { // value < 0
        colorVar = '--color-value-neg-bg';
        // Ensure minV is negative and significantly different from 0
        if (minV >= -epsilon) return getComputedStyle(document.documentElement).getPropertyValue('--color-value-zero-bg');
        // Normalize intensity based on negative range [minV, 0]
        const intensity = Math.min(1, Math.max(0, value / minV)); // value/minV is positive
        // Adjust alpha based on intensity
        const baseColor = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();
        const rgbaMatch = baseColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (rgbaMatch) {
            const originalAlpha = parseFloat(rgbaMatch[4] || '1');
            const newAlpha = Math.max(0.1, originalAlpha * intensity);
             return `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${newAlpha})`;
        }
         return baseColor; // Fallback
    }
}

// Add this helper function near the top:
function parseColorFromCSS(cssVar) {
    const colorStr = getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();
    
    // Try hex format first
    const hexMatch = colorStr.match(/#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})/);
    if (hexMatch) {
        return { r: parseInt(hexMatch[1], 16), g: parseInt(hexMatch[2], 16), b: parseInt(hexMatch[3], 16) };
    }
    
    // Try rgb format
    const rgbMatch = colorStr.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (rgbMatch) {
        return { r: parseInt(rgbMatch[1]), g: parseInt(rgbMatch[2]), b: parseInt(rgbMatch[3]) };
    }
    
    // Fallback
    return { r: 128, g: 128, b: 128 };
}

function adjustColorAlpha(cssVar, intensity) {
    const baseColor = getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();
    const rgbaMatch = baseColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (rgbaMatch) {
        const originalAlpha = parseFloat(rgbaMatch[4] || '1');
        const newAlpha = Math.max(0.1, originalAlpha * intensity);
        return `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${newAlpha})`;
    }
    return baseColor;
}

// Simplify interpolateProbColor function:
export function interpolateProbColor(prob) {
    const colorStart = parseColorFromCSS('--color-prob-zero');
    const colorEnd = parseColorFromCSS('--color-prob-one');
    const p = Math.max(0, Math.min(1, prob));
    
    const r = Math.round(colorStart.r + (colorEnd.r - colorStart.r) * p);
    const g = Math.round(colorStart.g + (colorEnd.g - colorStart.g) * p);
    const b = Math.round(colorStart.b + (colorEnd.b - colorStart.b) * p);
    
    return `rgb(${r}, ${g}, ${b})`;
}

// Simplify srValueToColor function:
function srValueToColor(value, maxM) {
    const epsilon = 1e-6;
    if (maxM < epsilon || value < epsilon) {
        return adjustColorAlpha('--color-sr-value-bg-min', 1.0);
    }
    const intensity = Math.min(1, Math.max(0, value / maxM));
    return adjustColorAlpha('--color-sr-value-bg-max', intensity);
}

// --- NEW: Function to draw the SR Vector M(hoveredState, :) ---
export function drawSRVector(ctx, gridSize, cellSize, hoveredStateKey, mTable, showSRText = true) {
    if (!hoveredStateKey || !mTable || !mTable[hoveredStateKey]) {
        // Draw nothing or a default background if no state is hovered or M-table is incomplete
        // For now, just return, the grid lines will show.
        return;
    }

    const srVector = mTable[hoveredStateKey];
    let maxM = -Infinity;
    let minM = Infinity; // Though typically non-negative, let's check

    // Find min/max in the specific vector M(hoveredState, :)
    for (let x_prime = 0; x_prime < gridSize; x_prime++) {
        for (let y_prime = 0; y_prime < gridSize; y_prime++) {
            const state_prime = `${x_prime},${y_prime}`;
            const value = srVector[state_prime] ?? 0;
            if (value > maxM) maxM = value;
            if (value < minM) minM = value;
        }
    }
    if (maxM === -Infinity) maxM = 0; // Handle case of all zeros
    if (minM === Infinity) minM = 0;

    // --- Text Style (Optional) ---
    const fontSize = Math.max(8, Math.floor(cellSize * 0.20));
    ctx.font = `${fontSize}px ${getComputedStyle(document.documentElement).getPropertyValue('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // --- Draw Cells ---
    for (let x_prime = 0; x_prime < gridSize; x_prime++) {
        for (let y_prime = 0; y_prime < gridSize; y_prime++) {
            const state_prime = `${x_prime},${y_prime}`;
            const mValue = srVector[state_prime] ?? 0;
            const color = srValueToColor(mValue, maxM);

            // Draw cell background color
            ctx.fillStyle = color;
            ctx.fillRect(x_prime * cellSize, y_prime * cellSize, cellSize, cellSize);

            // Draw SR value text if enabled
            if (showSRText) {
                const textX = x_prime * cellSize + cellSize / 2;
                const textY = y_prime * cellSize + cellSize / 2;

                // Use CSS variables for text color and shadow
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-sr-value-text');
                ctx.shadowColor = getComputedStyle(document.documentElement).getPropertyValue('--color-sr-value-text-shadow');
                ctx.shadowBlur = 2;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;

                ctx.fillText(mValue.toFixed(2), textX, textY);

                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;
            }
        }
    }
}

// Function to draw the value function V(s) or max Q(s,a) overlay
export function drawValues(ctx, gridSize, cellSize, qTable, vTable, mTable, wTable, currentAlgorithm, showValueText) {
    if (!qTable && !vTable && !mTable && !wTable) return;

    const { minV, maxV } = getMinMaxValues(gridSize, qTable, vTable, mTable, wTable, currentAlgorithm);

    // --- Text Style ---
    const fontSize = Math.max(8, Math.floor(cellSize * 0.25));
    ctx.font = `${fontSize}px ${getComputedStyle(document.documentElement).getPropertyValue('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const state = `${x},${y}`;
            const value = getValue(state, qTable, vTable, mTable, wTable, currentAlgorithm, gridSize);
            const color = valueToColor(value, minV, maxV);

            // Draw cell background color
            ctx.fillStyle = color;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);

            // Draw value text if enabled
            if (showValueText) {
                const textX = x * cellSize + cellSize / 2;
                const textY = y * cellSize + cellSize / 2;
                // Use CSS variables for text color and shadow
                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-value-text');
                ctx.shadowColor = getComputedStyle(document.documentElement).getPropertyValue('--color-value-text-shadow');
                ctx.shadowBlur = 2;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;

                ctx.fillText(value.toFixed(2), textX, textY);

                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;
            }
        }
    }
}

// Function to draw floating reward text
export function drawRewardText(ctx, text, pos, cellSize, alpha, offsetY) {
    if (alpha <= 0) return;

    const pixelX = (pos.x + 0.5) * cellSize;
    const pixelY = (pos.y + 0.5) * cellSize - offsetY;

    ctx.save(); // Save context state
    ctx.globalAlpha = alpha;
    ctx.font = 'bold 20px Arial'; // Font style
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';

    // Use CSS variables for color and shadow, respecting current alpha
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--color-reward-text');
    const shadowColor = getComputedStyle(document.documentElement).getPropertyValue('--color-reward-text-shadow');

    // Apply alpha to the text color
    const rgbaMatch = textColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
     if (rgbaMatch) {
        ctx.fillStyle = `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${alpha})`;
     } else {
        ctx.fillStyle = textColor; // Fallback if var isn't RGBA
     }

    ctx.shadowColor = shadowColor;
    ctx.shadowBlur = 5;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 1;

    ctx.fillText(text, pixelX, pixelY);
    ctx.restore(); // Restore context state (removes alpha, shadow, etc.)
}

// --- Image Loading ---
export function loadImages() {
    return new Promise((resolve, reject) => {
        let imagesPending = 4; // Increased to 4 for wall image

        function onImageLoad() {
            imagesPending--;
            if (imagesPending === 0) {
                console.log("Images loaded.");
                resolve(); // Resolve the promise when all images are loaded
            } else {
                console.log(`Waiting for ${imagesPending} more image(s) to load...`);
            }
        }

        function onImageError(imageName) {
             console.error(`Failed to load ${imageName} image.`);
             reject(new Error(`Failed to load ${imageName}`)); // Reject the promise on error
        }


        agentImage.onload = () => { console.log("Agent image loaded."); agentLoaded = true; onImageLoad(); };
        rewardImage.onload = () => { console.log("Reward image loaded."); rewardLoaded = true; onImageLoad(); };
        badStateImage.onload = () => { console.log("Bad state image loaded."); badStateLoaded = true; onImageLoad(); };
        wallImage.onload = () => { console.log("Wall image loaded."); wallLoaded = true; onImageLoad(); };

        agentImage.onerror = () => onImageError("agent");
        rewardImage.onerror = () => onImageError("reward");
        badStateImage.onerror = () => onImageError("bad state");
        wallImage.onerror = () => onImageError("wall");

        // Helper function to safely encode SVG strings for btoa
        const safeBtoa = (svgString) => btoa(unescape(encodeURIComponent(svgString)));

        agentImage.src = 'data:image/svg+xml;base64,' + safeBtoa(agentSvgString);
        rewardImage.src = 'data:image/svg+xml;base64,' + safeBtoa(rewardSvgString);
        badStateImage.src = 'data:image/svg+xml;base64,' + safeBtoa(badStateSvgString);
        wallImage.src = 'data:image/svg+xml;base64,' + safeBtoa(wallSvgString);
    });
}

// --- Agent Interaction ---
export function takeAction(action, currentAgentPos, gridSize) {
    let { x, y } = currentAgentPos;
    let reward = stepPenalty; // Use the configurable step penalty
    let potentialNextX = x;
    let potentialNextY = y;
    let done = false; // Initialize done flag

    switch (action) {
        case 'up':
            potentialNextY = Math.max(0, y - 1);
            break;
        case 'down':
            potentialNextY = Math.min(gridSize - 1, y + 1);
            break;
        case 'left':
            potentialNextX = Math.max(0, x - 1);
            break;
        case 'right':
            potentialNextX = Math.min(gridSize - 1, x + 1);
            break;
    }

    let nextX = x; // Start assuming no move
    let nextY = y;

    // Check if the potential next cell is within bounds and NOT a wall
    if (potentialNextX >= 0 && potentialNextX < gridSize &&
        potentialNextY >= 0 && potentialNextY < gridSize &&
        gridRewards[potentialNextY][potentialNextX] !== REWARD_WALL) {
        // If it's a valid move (not into a wall), update the next position
        nextX = potentialNextX;
        nextY = potentialNextY;
    } else {
        // If the move is invalid (out of bounds or into a wall),
        // the agent stays in the current position (x, y).
        // Optionally, apply a different penalty for hitting a wall
        // reward = -1.0; // Example: Larger penalty for hitting a wall
    }


    const newAgentPos = { x: nextX, y: nextY };
    const nextState = `${nextX},${nextY}`;
    const cellState = gridRewards[nextY][nextX]; // Check the state of the *actual* destination cell

    // Check reward based on the state of the destination cell
    // This logic only applies if the agent actually moved (nextX, nextY changed)
    // or if the agent started on a gem/bad state and tried to move off but couldn't (e.g., walled in).
    if (cellState === REWARD_GEM) {
        // reward = REWARD_MAGNITUDE_GEM; // Use constant // OLD
        reward = rewardMagnitudeGem; // MODIFIED: Use variable
        if (terminateOnGem) {
            done = true; // Set done flag if setting is enabled
        }
    } else if (cellState === REWARD_BAD) {
        // reward = REWARD_MAGNITUDE_BAD; // Use constant // OLD
        reward = rewardMagnitudeBad; // MODIFIED: Use variable
        done = true; // Episode always terminates on hitting the bad state
    }
    // If cellState is REWARD_EMPTY or REWARD_WALL (which agent can't enter, but might start on if forced?),
    // the step penalty applies (or wall penalty if added above).

    return { nextState, reward, newAgentPos, done }; // Return done flag
}

// --- Dynamic Reward Modification ---
export function cycleCellState(x, y, gridSize) {
    if (x < 0 || x >= gridSize || y < 0 || y >= gridSize) {
        console.error("Invalid cell coordinates:", x, y);
        return false; // Indicate failure
    }

    // Ensure the cell is not the current agent start position
    if (x === startPos.x && y === startPos.y) {
        console.warn("Cannot change the state of the agent's start cell.");
        return false;
    }

    const currentState = gridRewards[y][x];
    let nextState;

    if (currentState === REWARD_EMPTY) {
        nextState = REWARD_GEM; // Empty -> Gem
    } else if (currentState === REWARD_GEM) {
        nextState = REWARD_BAD;  // Gem -> Bad
    } else if (currentState === REWARD_BAD) {
        nextState = REWARD_WALL; // Bad -> Wall (NEW)
    } else { // currentState === REWARD_WALL
        nextState = REWARD_EMPTY; // Wall -> Empty
    }

    gridRewards[y][x] = nextState;
    console.log(`Cell (${x}, ${y}) state changed to ${nextState}`);
    return true; // Indicate success
}

// Helper to get the reward grid (e.g., for algorithms)
export function getGridRewards() {
    // Return a deep copy to prevent external modification
    return gridRewards.map(row => [...row]);
}

// Function to draw the policy arrows (greedy action) for each state
// Draws a single, large, centered ASCII arrow character based on the first best action found.
// Arrow color now based on action probability.
// Accepts relevant tables and the current algorithm
export function drawPolicyArrows(ctx, gridSize, cellSize, qTable, hTable, mTable, wTable, currentAlgorithm, takeActionFuncForEnvLookup, currentAgentPosForLookup) {
    if (!qTable && !hTable && !mTable && !wTable) return;

    // --- Style for the ASCII Arrows ---
    const fontSize = Math.max(12, Math.floor(cellSize * 0.6));
    ctx.font = `bold ${fontSize}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const state = `${x},${y}`;
            // Get agent position for this state (needed for SR Q lookups)
            const stateAgentPos = { x: x, y: y };
            // Pass lookup functions needed for SR Q-value calculation inside these helpers
            const bestActions = getBestActions(state, takeActionFuncForEnvLookup, stateAgentPos);
            const actionProbs = getActionProbabilities(state, takeActionFuncForEnvLookup, stateAgentPos);

            const centerX = x * cellSize + cellSize / 2;
            const centerY = y * cellSize + cellSize / 2;

            let charToDraw = '';
            let actionProb = 0;

            if (bestActions.length === 0 || !actions.some(a => actionProbs[a] > 0)) {
                charToDraw = '●';
                actionProb = 0.5; // Default grey
            } else {
                const actionToDraw = bestActions[0];
                actionProb = actionProbs[actionToDraw] || (1.0 / actions.length);

                switch (actionToDraw) {
                    case 'up':    charToDraw = '↑'; break;
                    case 'down':  charToDraw = '↓'; break;
                    case 'left':  charToDraw = '←'; break;
                    case 'right': charToDraw = '→'; break;
                    default:      charToDraw = '?';
                }
            }

            ctx.fillStyle = interpolateProbColor(actionProb);
            ctx.fillText(charToDraw, centerX, centerY);
        }
    }
}

export function drawSRWVector(ctx, gridSize, cellSize, wTable, showWText = true) {
    if (!wTable) {
        return;
    }

    let maxW = -Infinity;
    let minW = Infinity;

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const state = `${x},${y}`;
            const value = wTable[state] ?? 0;
            if (value > maxW) maxW = value;
            if (value < minW) minW = value;
        }
    }
    if (maxW === -Infinity) maxW = 0;
    if (minW === Infinity) minW = 0;

    const fontSize = Math.max(8, Math.floor(cellSize * 0.20));
    ctx.font = `${fontSize}px ${getComputedStyle(document.documentElement).getPropertyValue('--font-mono') || 'monospace'}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const state = `${x},${y}`;
            const wValue = wTable[state] ?? 0;
            
            const color = wValueToColor(wValue, minW, maxW);

            ctx.fillStyle = color;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);

            if (showWText) {
                const textX = x * cellSize + cellSize / 2;
                const textY = y * cellSize + cellSize / 2;

                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--color-sr-value-text');
                ctx.shadowColor = getComputedStyle(document.documentElement).getPropertyValue('--color-sr-value-text-shadow');
                ctx.shadowBlur = 2;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;

                ctx.fillText(wValue.toFixed(2), textX, textY);

                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;
            }
        }
    }
}

function wValueToColor(value, minW, maxW) {
    const epsilon = 1e-6;

    if (Math.abs(maxW - minW) < epsilon || Math.abs(value) < epsilon) {
        return getComputedStyle(document.documentElement).getPropertyValue('--color-value-zero-bg');
    }

    let colorVar;

    if (value > 0) {
        colorVar = '--color-value-pos-bg';
        if (maxW <= epsilon) return getComputedStyle(document.documentElement).getPropertyValue('--color-value-zero-bg');
        const intensity = Math.min(1, Math.max(0, value / maxW));
        const baseColor = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();
        const rgbaMatch = baseColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (rgbaMatch) {
            const originalAlpha = parseFloat(rgbaMatch[4] || '1');
            const newAlpha = Math.max(0.1, originalAlpha * intensity);
            return `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${newAlpha})`;
        }
        return baseColor;
    } else {
        colorVar = '--color-value-neg-bg';
        if (minW >= -epsilon) return getComputedStyle(document.documentElement).getPropertyValue('--color-value-zero-bg');
        const intensity = Math.min(1, Math.max(0, value / minW));
        const baseColor = getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim();
        const rgbaMatch = baseColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (rgbaMatch) {
            const originalAlpha = parseFloat(rgbaMatch[4] || '1');
            const newAlpha = Math.max(0.1, originalAlpha * intensity);
            return `rgba(${rgbaMatch[1]}, ${rgbaMatch[2]}, ${rgbaMatch[3]}, ${newAlpha})`;
        }
        return baseColor;
    }
}
