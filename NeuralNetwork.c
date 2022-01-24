#include "NeuralNetwork.h"
#include <assert.h>
#include <math.h>

Result const c_EMPTY_RESULT = {
    .m_Size = 0,
    .m_Values = { 0 }
};

/* Internal memory helpers
/*----------------------------*/
void* LNN_MemoryBufferMalloc(MemoryBuffer* buffer, u32 allocSizeInBytes) {
    if (buffer->m_UsedSize + allocSizeInBytes > buffer->m_MaxSize) {
        printf("MEMORY OVERFLOW! What to heck\n");
        return NULL;
    }

    char* ptr = buffer->m_Memory + buffer->m_UsedSize;
    buffer->m_UsedSize += allocSizeInBytes;
    return ptr;
}

MemoryBuffer LNN_MemoryBufferCreate(u32 size) {
    MemoryBuffer newMemBuffer = {
        .m_MaxSize = size,
        .m_UsedSize = 0,
        .m_Memory = malloc(size)
    };
    return newMemBuffer;
}

/* Activation functions 
/*----------------------------*/
NodeValue Sigmoid(NodeValue x) {
    return 1.0f / (1.0f + exp(-x));
}

NodeValue DerivativeSigmoid(NodeValue x) {
    return Sigmoid(x) * (1.0f - Sigmoid(x));
}

/* Network library
/*----------------------------*/
// Define the network's structure, allocate all memory and get a usable struct
Network* LNN_CreateNetwork(NetworkSettings* settings) {
    Network* network = LNN_AllocateNetwork(settings->m_Size, settings->m_LayerSizes);
    
    network->m_LearningRate = settings->m_LearningRate;
    network->m_Momentum = settings->m_Momentum;

    return network;
}

Network* LNN_AllocateNetwork(u32 numLayers, u32* layerSizes) {
    // Calculate the memory size of the network
    u32 totalNumNodes = 0;
    for (size_t i = 0; i < numLayers; i++) {
        totalNumNodes += layerSizes[i];
    }

    u32 totalNumSynapses = 0;
    for (size_t i = 1; i < numLayers; i++) {
        totalNumSynapses += layerSizes[i] * layerSizes[i - 1];
    }

    u32 nodeLayersSize = numLayers * sizeof(NodeLayer);
    u32 nodesSize = totalNumNodes * sizeof(Node);
    u32 synapsesSize = totalNumSynapses * sizeof(Synapse);
    u32 totalSize = nodeLayersSize + nodesSize + synapsesSize + sizeof(Network);

    MemoryBuffer buffer = LNN_MemoryBufferCreate(totalSize);

    // Create network
    Network* network = LNN_MemoryBufferMalloc(&buffer, sizeof(Network));
    network->m_NumLayers = numLayers;
    network->m_Memory = buffer;
    network->m_NodeLayers = LNN_MemoryBufferMalloc(&network->m_Memory, sizeof(NodeLayer) * numLayers);

    // Init node layers
    for (size_t layerIndex = 0; layerIndex < numLayers; layerIndex++) {
        network->m_NodeLayers[layerIndex].m_Size = layerSizes[layerIndex];
        network->m_NodeLayers[layerIndex].m_Nodes = LNN_MemoryBufferMalloc(&network->m_Memory, sizeof(Node) * layerSizes[layerIndex]);

        NodeLayer layer = network->m_NodeLayers[layerIndex];

        // Create nodes synapses
        bool isFirstLayer = (layerIndex == 0);
        for (size_t nodeIndex = 0; nodeIndex < layer.m_Size; nodeIndex++) {
            if (isFirstLayer) {
                layer.m_Nodes[nodeIndex].m_SynapseCount = 0; // Input layer has 0 synapses
                layer.m_Nodes[nodeIndex].m_Synapses = NULL;
            }
            else {
                u32 synapseCount = network->m_NodeLayers[layerIndex - 1].m_Size;
                layer.m_Nodes[nodeIndex].m_SynapseCount = synapseCount;
                layer.m_Nodes[nodeIndex].m_Synapses = LNN_MemoryBufferMalloc(&network->m_Memory, sizeof(Synapse) * synapseCount);
            }

            layer.m_Nodes[nodeIndex].m_Value = 0.0;
        }
    }

    // Zero out synapse deltas and randomise all weights
    for (size_t layerIndex = 1; layerIndex < network->m_NumLayers; layerIndex++) {
        for (size_t nodeIndex = 0; nodeIndex < network->m_NodeLayers[layerIndex].m_Size; nodeIndex++) {
            Node node = network->m_NodeLayers[layerIndex].m_Nodes[nodeIndex];
            for (size_t synIndex = 0; synIndex < node.m_SynapseCount; synIndex++) {
                Weight x = ((Weight)rand() / (Weight)(RAND_MAX / 1.0) - 0.5f) * 2.0f;
                node.m_Synapses[synIndex].m_Delta = 0.0f;
                node.m_Synapses[synIndex].m_Weight = x;
            }
        }
    }

    return network;
}

void LNN_FreeNetwork(Network* network) {
    free(network->m_Memory.m_Memory);
}

// Simple input/output, forward pass
Result LNN_ForwardPropagate(Network* network, Input input) {
    // Make sure input size is same as network first layer
    if (input.m_Size != network->m_NodeLayers[0].m_Size) {
        perror("Input size is not the same size as the first layer of nodes in this network.\n");
        return c_EMPTY_RESULT;
    }

    // Copy input to first node layer
    for (size_t i = 0; i < network->m_NodeLayers[0].m_Size; i++) {
        network->m_NodeLayers[0].m_Nodes[i].m_Value = input.m_Values[i];
    }

    // Forward propagate throughout network
    for (size_t layerIndex = 1; layerIndex < network->m_NumLayers; layerIndex++) {
        NodeLayer previousLayer = network->m_NodeLayers[layerIndex - 1];
        NodeLayer layer = network->m_NodeLayers[layerIndex];

        for (size_t nodeIndex = 0; nodeIndex < layer.m_Size; nodeIndex++) {
            Node* currentNode = &layer.m_Nodes[nodeIndex];
            currentNode->m_PreActivatedValue = 0.0;

            // Add up all previous nodes values *connecting synapse
            for (size_t synapseIndex = 0; synapseIndex < currentNode->m_SynapseCount; synapseIndex++) {
                currentNode->m_PreActivatedValue += currentNode->m_Synapses[synapseIndex].m_Weight * previousLayer.m_Nodes[synapseIndex].m_Value;
            }

            // Pass through activation function
            currentNode->m_Value = Sigmoid(currentNode->m_PreActivatedValue);
        }
    }

    // Create result
    NodeLayer lastLayer = network->m_NodeLayers[network->m_NumLayers - 1];
    Result result = c_EMPTY_RESULT;
    result.m_Size = lastLayer.m_Size;
    for (size_t i = 0; i < lastLayer.m_Size; i++) {
        result.m_Values[i] = lastLayer.m_Nodes[i].m_Value;
    }

    return result;
}

void LNN_SetSynapseWeight(Network* network, u32 layer, u32 node, u32 synapse, Weight weight) {
    network->m_NodeLayers[layer].m_Nodes[node].m_Synapses[synapse].m_Weight = weight;
}

// Algorithms from https://www.youtube.com/watch?v=tIeHLnjs5U8
void BackpropagateOutputLayer(Network* network, Input input, Result expectedResult) {
    NodeLayer currentLayer = network->m_NodeLayers[network->m_NumLayers - 1];
    NodeLayer previousLayer = network->m_NodeLayers[network->m_NumLayers - 2];

    for (size_t i = 0; i < currentLayer.m_Size; i++)
    {
        Node currentNode = currentLayer.m_Nodes[i];
        for (size_t j = 0; j < currentNode.m_SynapseCount; j++)
        {
            f64 c0aL = 2 * (currentNode.m_Value - expectedResult.m_Values[i]);
            f64 aLzL = DerivativeSigmoid(currentNode.m_PreActivatedValue);
            f64 zLwL = previousLayer.m_Nodes[j].m_Value;
            f64 derivative = c0aL * aLzL * zLwL;
            currentNode.m_Synapses[j].m_Delta = derivative + currentNode.m_Synapses[j].m_Delta * network->m_Momentum;
        }
    }
}

void BackpropagateHiddenLayer(Network* network, Input input, Result expectedResult, int L) {
    NodeLayer currentLayer = network->m_NodeLayers[L];
    NodeLayer previousLayer = network->m_NodeLayers[L - 1];
    NodeLayer forwardLayer = network->m_NodeLayers[L + 1];

    for (size_t nodeIndex = 0; nodeIndex < currentLayer.m_Size; nodeIndex++)
    {
        Node currentNode = currentLayer.m_Nodes[nodeIndex];
        for (size_t j = 0; j < currentNode.m_SynapseCount; j++)
        {
            f64 weightedSumInfluence = 0.0f;
            for (size_t k = 0; k < forwardLayer.m_Size; k++)
            {
                assert(nodeIndex < forwardLayer.m_Nodes[k].m_SynapseCount);
                weightedSumInfluence += forwardLayer.m_Nodes[k].m_Synapses[nodeIndex].m_Delta * forwardLayer.m_Nodes[k].m_Synapses[nodeIndex].m_Weight;
            }

            f64 c0aL = (weightedSumInfluence / (f64)forwardLayer.m_Size);
            f64 aLzL = DerivativeSigmoid(currentNode.m_PreActivatedValue);
            f64 zLwL = previousLayer.m_Nodes[j].m_Value;
            f64 derivative = c0aL * aLzL * zLwL;
            currentNode.m_Synapses[j].m_Delta = derivative + currentNode.m_Synapses[j].m_Delta * network->m_Momentum;
        }
    }
}


f64 LNN_Learn(Network* network, Input input, Result expectedResult) {
    // (´⊙ω⊙`)!!! Ohgod

    // Forward prop it to cache node values
    Result actualResult = LNN_ForwardPropagate(network, input);

    BackpropagateOutputLayer(network, input, expectedResult);

    for (int i = network->m_NumLayers - 2; i > 0; i--)
    {
        BackpropagateHiddenLayer(network, input, expectedResult, i);
    }

    // Apply negative derivatives to weights so that the error value becomes smaller
    for (size_t layerIndex = 1; layerIndex < network->m_NumLayers; layerIndex++) {
        for (size_t nodeIndex = 0; nodeIndex < network->m_NodeLayers[layerIndex].m_Size; nodeIndex++) {
            Node node = network->m_NodeLayers[layerIndex].m_Nodes[nodeIndex];
            for (size_t synIndex = 0; synIndex < node.m_SynapseCount; synIndex++) {
                node.m_Synapses[synIndex].m_Weight -= node.m_Synapses[synIndex].m_Delta * network->m_LearningRate;
            }
        }
    }

    // Returns the MSE
    f64 mse = 0.0f;
    for (size_t i = 0; i < actualResult.m_Size; i++)
    {
        f64 e = actualResult.m_Values[i] - expectedResult.m_Values[i];
        mse += e * e;
    }
    return mse / (f64)actualResult.m_Size;
}
