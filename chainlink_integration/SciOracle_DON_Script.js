const problem = args[0];
const solution = args[1];

// Make the HTTP Request directly to the SciOracle Core GPU Node
const url = "https://your-scioracle-domain.com/api/chainlink/verify";

console.log(`Sending Discovery to SciOracle Matrix: ${problem} => ${solution}`);

const apiResponse = await Functions.makeHttpRequest({
    url: url,
    method: "POST",
    timeout: 9000, // Important: Chainlink Functions limit is 10s
    headers: {
        "Content-Type": "application/json",
        // "Authorization": `Bearer ${secrets.API_KEY}` // Mount securely via DON Secrets
    },
    data: {
        problem: problem,
        solution: solution
    }
});

if (apiResponse.error) {
    throw Error(`SciOracle API Runtime Error: ${apiResponse.message}`);
}

const data = apiResponse.data;

if (!data || data.error) {
    throw Error(`Mathematical processing failed: ${data?.error || "Unknown"}`);
}

// Extract variables from the MathEBM
const energy = data.energy; // float between 0.0 and 1.0
const isSound = data.is_sound ? 1 : 0; // boolean converted to uint8

// Scale the floating point energy tensor into an integer (e.g., 0.012 -> 12000000000000000n)
const scaledEnergy = BigInt(Math.floor(energy * 10 ** 18));

console.log(`SciOracle Eval Complete | Energy: ${scaledEnergy.toString()} | Sound: ${isSound}`);

// Safely pack the bytes using Ethers ABI encoding so Solidity decode logic succeeds
const encoded = ethers.utils.defaultAbiCoder.encode(
    ["uint256", "uint8"],
    [scaledEnergy, isSound]
);

// We slice off the 0x prefix and return Buffer
return Buffer.from(encoded.slice(2), "hex");
