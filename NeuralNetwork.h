#include <stdio.h>
#include <stdlib.h>  

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
} Synapse;

typedef struct {
	// Witholds synapses that connect to the previous layer
	u32 m_SynapseCount; 
	Synapse* m_Synapses;
	NodeValue m_Value;
} Node;

typedef struct {
	u32 m_Size;
	Node* m_Nodes;
} NodeLayer;

typedef struct {
	u32 m_NumLayers;
	MemoryBuffer m_Memory;
	NodeLayer* m_NodeLayers;
} Network;

typedef struct {
	u32 m_Size;
	NodeValue m_Values[30]; // Alloc 30 node values, that's the max result size!
} Result;

typedef Result Input;

Network* CreateNetwork(u32 numLayers, u32* layerSizes);
void FreeNetwork(Network* network);
Network* AllocateNetwork(u32 numLayers, u32* layerSizes);
Result ForwardPropagate(Network* network, Input input);
void BackPropagate(Network* network, Input input, Result result);