#include "NeuralNetwork.h"

#include <math.h>

Result const c_EMPTY_RESULT = {
    .m_Size = 0,
    .m_Values = { 0 }
};

/* Internal memory helpers
/*----------------------------*/
void* MemoryBufferMalloc(MemoryBuffer* buffer, u32 allocSizeInBytes) {
    if (buffer->m_UsedSize + allocSizeInBytes > buffer->m_MaxSize) {
        printf("MEMORY OVERFLOW! What to heck\n");
        return NULL;
    }

    char* ptr = buffer->m_Memory + buffer->m_UsedSize;
    buffer->m_UsedSize += allocSizeInBytes;
    return ptr;
}

MemoryBuffer MemoryBufferCreate(u32 size) {
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
    return 1.0 / (1.0 + exp(-x));
}

NodeValue Relu(NodeValue x) {
    return fmax(x, 0.0);
}

/* Network library
/*----------------------------*/
// Define the network's structure, allocate all memory and get a usable struct
// e.g. CreateNetwork(3, {2, 3, 1});
Network* CreateNetwork(NetworkSettings* settings) {
    Network* network = AllocateNetwork(settings->m_Size, settings->m_LayerSizes);
    
    network->m_ActivationFunction = settings->m_ActivationFunction;
    
    // Extra network initialisation here

    return network;
}

Network* AllocateNetwork(u32 numLayers, u32* layerSizes) {
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
    u32 totalSize = nodeLayersSize + nodesSize + synapsesSize;

    // Create network
    Network* network = malloc(sizeof(Network));
    if (network == NULL) {
        printf("Cannot malloc network\n");
        exit(-1);
    }

    network->m_NumLayers = numLayers;
    network->m_Memory = MemoryBufferCreate(totalSize);
    network->m_NodeLayers = MemoryBufferMalloc(&network->m_Memory, sizeof(NodeLayer) * numLayers);

    // Init node layers
    for (size_t layerIndex = 0; layerIndex < numLayers; layerIndex++) {
        network->m_NodeLayers[layerIndex].m_Size = layerSizes[layerIndex];
        network->m_NodeLayers[layerIndex].m_Nodes = MemoryBufferMalloc(&network->m_Memory, sizeof(Node) * layerSizes[layerIndex]);

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
                layer.m_Nodes[nodeIndex].m_Synapses = MemoryBufferMalloc(&network->m_Memory, sizeof(Synapse) * synapseCount);
            }

            layer.m_Nodes[nodeIndex].m_Value = 0.0;
        }
    }

    return network;
}

void FreeNetwork(Network* network) {
    // Free internal memory buffer
    free(network->m_Memory.m_Memory);
    // Free network itself
    free(network);
}

// Simple input/output, forward pass
Result ForwardPropagate(Network* network, Input input) {
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
            currentNode->m_Value = 0.0;

            // Add up all previous nodes values *connecting synapse
            for (size_t synapseIndex = 0; synapseIndex < currentNode->m_SynapseCount; synapseIndex++) {
                currentNode->m_Value += currentNode->m_Synapses[synapseIndex].m_Weight * previousLayer.m_Nodes[synapseIndex].m_Value;
            }

            // "Activate" node value
            switch (network->m_ActivationFunction) {
            case ACTIVATION_FUNCTION_RELU: 
                currentNode->m_Value = Relu(currentNode->m_Value);
                break;
            case ACTIVATION_FUNCTION_SIGMOID:
                currentNode->m_Value = Sigmoid(currentNode->m_Value);
                break;
            default: break;
            }
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

void SetSynapseWeight(Network* network, u32 layer, u32 node, u32 synapse, Weight weight) {
    network->m_NodeLayers[layer].m_Nodes[node].m_Synapses[synapse].m_Weight = weight;
}

// Learn
void BackPropagate(Network* network, Input input, Result result) {
    // (´⊙ω⊙`)!!! oh god
}