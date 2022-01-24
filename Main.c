#include <stdio.h>
#include <stdarg.h>
#include "NeuralNetwork.h"

bool FloatClose(NodeValue input, NodeValue desired, NodeValue epsilon) {
    NodeValue val = (input - desired);
    if (val < 0.0f) val = -val;
    return val < epsilon;
}

int main() {
    NetworkSettings settings = {
        .m_Size = 3,
        .m_LayerSizes = {2,5,1},
        .m_LearningRate = 0.4,
        .m_Momentum = 0.0,
    };
    Network* network = LNN_CreateNetwork(&settings);

    Input inputA = { .m_Size = 2, .m_Values = { 0.0, 0.0 } };
    Input inputB = { .m_Size = 2, .m_Values = { 1.0, 0.0 } };
    Input inputC = { .m_Size = 2, .m_Values = { 0.0, 1.0 } };
    Input inputD = { .m_Size = 2, .m_Values = { 1.0, 1.0 } };
    Result outputA = { .m_Size = 1, .m_Values = { 0.0 } };
    Result outputB = { .m_Size = 1, .m_Values = { 1.0 } };
    Result outputC = { .m_Size = 1, .m_Values = { 1.0 } };
    Result outputD = { .m_Size = 1, .m_Values = { 0.0 } };

    printf("Beginning learning...\n");

    f64 e = 0.0;
    f64 c = 0.0;
    u32 numIterations = 50000;
    for (size_t i = 0; i < numIterations; i++)
    {
        e += LNN_Learn(network, inputA, outputA);
        e += LNN_Learn(network, inputB, outputB);
        e += LNN_Learn(network, inputC, outputC);
        e += LNN_Learn(network, inputD, outputD);
        c += 4.0;
        if (i % (numIterations/10) == 0) {
            printf("Error: %f\n", e / c);
            e = 0.0;
            c = 0.0;
        }
    }

    printf("%f, %f -> %f\n%f, %f -> %f\n%f, %f -> %f\n%f, %f -> %f\n", 
        inputA.m_Values[0], inputA.m_Values[1], LNN_ForwardPropagate(network, inputA).m_Values[0],
        inputB.m_Values[0], inputB.m_Values[1], LNN_ForwardPropagate(network, inputB).m_Values[0],
        inputC.m_Values[0], inputC.m_Values[1], LNN_ForwardPropagate(network, inputC).m_Values[0],
        inputD.m_Values[0], inputD.m_Values[1], LNN_ForwardPropagate(network, inputD).m_Values[0]
    );

    bool successful =
        FloatClose(LNN_ForwardPropagate(network, inputA).m_Values[0], outputA.m_Values[0], 0.1) &&
        FloatClose(LNN_ForwardPropagate(network, inputB).m_Values[0], outputB.m_Values[0], 0.1) &&
        FloatClose(LNN_ForwardPropagate(network, inputC).m_Values[0], outputC.m_Values[0], 0.1) &&
        FloatClose(LNN_ForwardPropagate(network, inputD).m_Values[0], outputD.m_Values[0], 0.1);

    if (successful) {
        printf("\x1b[32mTests pass! :D\x1b[0m\n");
    }
    else {
        printf("\x1b[31mTests fail! :(\x1b[0m\n");
    }

    LNN_FreeNetwork(network);
}