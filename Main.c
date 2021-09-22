#include <stdio.h>

#include "NeuralNetwork.h"

int main() {
    u32 sizes[3] = { 2, 2, 1 };
    Network* network = CreateNetwork(3, sizes);

    /*
     * I1->A = 1.0
     * I2->A = -1.0
     * I1->B = -1.0
     * I2->B = 1.0
     * A->O = 1.0
     * B->O = 1.0
     * 
     *   I1 - A \
     *      X    O
     *   I2 - B /
     */

     // Manually assign the network's synapse weights to make a simple XOR network
     // Node A
    network->m_NodeLayers[1].m_Nodes[0].m_Synapses[0].m_Weight = 10.0;
    network->m_NodeLayers[1].m_Nodes[0].m_Synapses[1].m_Weight = -10.0;

    // Node B
    network->m_NodeLayers[1].m_Nodes[1].m_Synapses[0].m_Weight = -10.0;
    network->m_NodeLayers[1].m_Nodes[1].m_Synapses[1].m_Weight = 10.0;

    // Final output node
    network->m_NodeLayers[2].m_Nodes[0].m_Synapses[0].m_Weight = 10.0;
    network->m_NodeLayers[2].m_Nodes[0].m_Synapses[1].m_Weight = 10.0;

    Input input = {
        .m_Size = 2,
        .m_Values = {
            1.0,
            0.0
        },
    };

    Result result = ForwardPropogate(network, input);

    for (size_t layerIndex = 0; layerIndex < network->m_NumLayers; layerIndex++) {
        for (size_t nodeIndex = 0; nodeIndex < network->m_NodeLayers[layerIndex].m_Size; nodeIndex++) {
            Node node = network->m_NodeLayers[layerIndex].m_Nodes[nodeIndex];
            printf("\tNode %llu %llu: %.4lf\n", layerIndex, nodeIndex, node.m_Value);
            // Print synapses 
            for (size_t synIndex = 0; synIndex < node.m_SynapseCount; synIndex++) {
                printf("\t\tSynapse %llu: %.4lf\n", synIndex, node.m_Synapses[synIndex].m_Weight);
            }
        }
    }

    printf("Result:\n");
    for (size_t i = 0; i < result.m_Size; i++) {
        printf("\t%llu: %.3lf\n", i, result.m_Values[i]);
    }

    FreeNetwork(network);
}