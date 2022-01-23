#include <stdio.h>
#include <stdlib.h>  

#define MAX_NODES_PER_LAYER 50
#define MAX_NUM_LAYERS 20

typedef unsigned int u32;
typedef int s32;
typedef long int s64;
typedef float f32;
typedef double f64;
typedef u32 bool;
typedef f64 Weight;
typedef f64 NodeValue;

typedef struct {
	char* m_Memory;
	u32 m_UsedSize;
	u32 m_MaxSize;
} MemoryBuffer;

typedef struct {
	Weight m_Weight;
	Weight m_Delta;
} Synapse;

typedef struct {
	// Witholds synapses that connect to the previous layer
	u32 m_SynapseCount; 
	Synapse* m_Synapses;
	NodeValue m_Value;
	NodeValue m_PreActivatedValue;
} Node;

typedef struct {
	u32 m_Size;
	Node* m_Nodes;
} NodeLayer;

typedef struct {
	f64 m_Momentum;
	f64 m_LearningRate;
	u32 m_NumLayers;
	MemoryBuffer m_Memory;
	NodeLayer* m_NodeLayers;
} Network;

typedef struct {
	u32 m_Size;
	NodeValue m_Values[MAX_NODES_PER_LAYER]; // Alloc max num of node values! Just so we can stack-allocate results & inputs
} NodeBuffer;

typedef struct {
	u32 m_Size;
	u32 m_LayerSizes[MAX_NUM_LAYERS];
	f64 m_LearningRate;
	f64 m_Momentum;
} NetworkSettings;


typedef NodeBuffer Input;
typedef NodeBuffer Result;

Network* LNN_CreateNetwork(NetworkSettings* settings);
void LNN_FreeNetwork(Network* network);
Network* LNN_AllocateNetwork(u32 numLayers, u32* layerSizes);
Result LNN_ForwardPropagate(Network* network, Input input);
f64 LNN_Learn(Network* network, Input input, Result expectedResult);
void LNN_SetSynapseWeight(Network* network, u32 layer, u32 node, u32 synapse, Weight weight);