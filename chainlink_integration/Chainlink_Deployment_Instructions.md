# SciOracle Chainlink Functions Integration

This directory contains the secure Smart Contract infrastructure required to formally mint "Proof of Discovery" (PoD) tokens on Ethereum/EVM chains utilizing the off-chain SciOracle Engine globally. 

We utilize **Chainlink Functions**, a serverless developer platform allowing smart contracts to run custom Javascript logic off-chain through a Decentralized Oracle Network (DON).

## Architecture

1. **The Core Validator (`SciOraclePoD.sol`)**:
   A Chainlink `FunctionsClient` and `ERC20` contract. Users call `submitDiscovery(problem, solution)`. The contract issues an event alerting the Chainlink Decentralized Oracle Network.
2. **The Oracle Logic (`SciOracle_DON_Script.js`)**:
   The Chainlink DON executes this Javascript. It fires an HTTP POST to your standalone SciOracle Node (`https://your-domain/api/chainlink/verify`).
3. **The Local Node Engine (`backend/app.py` -> `/api/chainlink/verify`)**:
   We added a high-priority, secure, 9-second-limited endpoint into the FastAPI backend specifically to calculate EBM Energy gradients and SymPy logic trees instantly for Oracles, skipping the heavier SQLite disk writes.
4. **The Fulfillment Callback (`SciOraclePoD.sol` -> `fulfillRequest`)**:
   The DON feeds the encoded bytes ABI (`uint256 energy, uint8 isSound`) back on-chain. The contract evaluates the metrics against the `targetDifficulty` and cryptographically mints `100 POD` coins if criteria are met.

---

## 🚀 Deployment Guide

### Step 1: Subscriptions & Funding
Because this uses Chainlink Functions, you must have LINK to fund the oracle requests.
1. Head to the [Chainlink Functions UI](https://functions.chain.link/).
2. Create a new "Subscription" on your desired network (e.g., Polygon Amoy, Sepolia, Base Sepolia).
3. Fund your subscription with `LINK` tokens.

### Step 2: Deploying the Smart Contract
Deploy `SciOraclePoD.sol` via Remix IDE, Hardhat, or Foundry.
Provide the constructor arguments depending on your network:
- `router`: E.g., `0x65Dcc24F8ff9e51F10DCc7Ed1e4e2A61e6E14bd6` (Polygon Amoy Router)
- `_donId`: E.g., `fun-polygon-amoy-1` (formatted as a bytes32)
- `_subscriptionId`: The integer ID of your subscription created in Step 1.
- `_sourceCode`: The exact plaintext payload of `SciOracle_DON_Script.js`.

### Step 3: Registering the Contract
Once deployed, go back to the [Chainlink Functions UI](https://functions.chain.link/) and add your smart contract's address as an **Authorized Consumer** to your subscription.

### Step 4: Configure SciOracle Backend Security
Your SciOracle node needs a static public IP / Domain Name for the Chainlink nodes to reach it. 
Inside `SciOracle_DON_Script.js`, change:
```javascript
const url = "https://your-scioracle-domain.com/api/chainlink/verify";
```
For deep security against DDoS, it is recommended to set a static `x-api-key` in `backend/app.py` and inject your `apiKey` directly into the Chainlink DON using [Chainlink Functions Secrets](https://docs.chain.link/chainlink-functions/resources/secrets). You then uncomment the Authorization header in the Javascript payload.

### Step 5: Mine Your First Discovery
Send a transaction directly to the contract:
```solidity
submitDiscovery("force equals mass times acceleration", "F = m*a");
```
Watch the transaction! Within ~10-15 seconds, the Chainlink Oracles will aggregate consensus against your server and trigger `fulfillRequest()`. If your node validates it via SymPy and rewards an Energy cost underneath the block limit, `100 POD` are instantly minted into your wallet address!
