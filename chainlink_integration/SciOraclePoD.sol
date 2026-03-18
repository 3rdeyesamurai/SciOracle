// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {FunctionsClient} from "@chainlink/contracts/src/v0.8/functions/dev/v1_0_0/FunctionsClient.sol";
import {FunctionsRequest} from "@chainlink/contracts/src/v0.8/functions/dev/v1_0_0/libraries/FunctionsRequest.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title SciOraclePoD
 * @notice A Chainlink Functions-enabled smart contract that mints "Discovery Coins" 
 * to users who submit mathematically sound, low-energy discoveries validated by the off-chain SciOracle EBM.
 */
contract SciOraclePoD is FunctionsClient, ERC20, Ownable {
    using FunctionsRequest for FunctionsRequest.Request;

    bytes32 public donId;
    uint64 public subscriptionId;
    string public sourceCode; // The JS code to run on the Chainlink DON
    uint32 public gasLimit = 300000;

    // Difficulty target (e.g., scaled by 1e18. 0.05 energy = 5e16)
    uint256 public targetDifficulty = 2e16; // Energy < 0.02
    uint256 public blockReward = 100 * 10**decimals();

    struct DiscoveryRequest {
        address miner;
        string problem;
        string solution;
    }

    mapping(bytes32 => DiscoveryRequest) public pendingRequests;
    mapping(bytes32 => bool) public verifiedSignatures;

    event DiscoverySubmitted(bytes32 indexed requestId, address indexed miner);
    event DiscoveryVerified(bytes32 indexed requestId, address indexed miner, uint256 energy, bool success);

    constructor(
        address router, 
        bytes32 _donId,
        uint64 _subscriptionId,
        string memory _sourceCode
    ) FunctionsClient(router) ERC20("Proof of Discovery", "POD") Ownable() {
        donId = _donId;
        subscriptionId = _subscriptionId;
        sourceCode = _sourceCode;
    }

    /**
     * @notice Submit a mathematical discovery to the network.
     * @param problem The natural language problem or equation.
     * @param solution The proposed mathematical solution.
     */
    function submitDiscovery(string calldata problem, string calldata solution) external returns (bytes32) {
        bytes32 signature = keccak256(abi.encodePacked(problem, solution));
        require(!verifiedSignatures[signature], "Discovery already exists on-chain");

        FunctionsRequest.Request memory req;
        req.initializeRequestForInlineJavaScript(sourceCode);
        
        string[] memory args = new string[](2);
        args[0] = problem;
        args[1] = solution;
        req.setArgs(args);

        bytes32 requestId = _sendRequest(
            req.encodeCBOR(),
            subscriptionId,
            gasLimit,
            donId
        );

        pendingRequests[requestId] = DiscoveryRequest({
            miner: msg.sender,
            problem: problem,
            solution: solution
        });

        emit DiscoverySubmitted(requestId, msg.sender);
        return requestId;
    }

    /**
     * @notice Callback invoked by Chainlink DON with the SciOracle EBM evaluation result.
     * @param requestId The ID of the request.
     * @param response Standardized bytes containing (uint256 energy_scaled_1e18, uint8 isSound).
     * @param err Any execution error from the DON.
     */
    function fulfillRequest(bytes32 requestId, bytes memory response, bytes memory err) internal override {
        DiscoveryRequest memory req = pendingRequests[requestId];
        require(req.miner != address(0), "Request not found");

        if (err.length > 0) {
            emit DiscoveryVerified(requestId, req.miner, 0, false);
            delete pendingRequests[requestId];
            return;
        }

        // Decode the Chainlink Function response
        // Expected format: abi.encode(uint256 energy, uint8 isSound)
        (uint256 energyScaled, uint8 isSound) = abi.decode(response, (uint256, uint8));

        bool success = false;
        if (isSound == 1 && energyScaled <= targetDifficulty) {
            // MINT THE REWARD!
            _mint(req.miner, blockReward);
            
            bytes32 signature = keccak256(abi.encodePacked(req.problem, req.solution));
            verifiedSignatures[signature] = true;
            success = true;
        }

        emit DiscoveryVerified(requestId, req.miner, energyScaled, success);
        delete pendingRequests[requestId];
    }
    
    // Admin functions
    function updateDifficulty(uint256 newDifficulty) external onlyOwner {
        targetDifficulty = newDifficulty;
    }

    function updateSourceCode(string calldata newSource) external onlyOwner {
        sourceCode = newSource;
    }
}
