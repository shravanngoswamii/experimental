// Define the specific types of nodes in a BUGS model
export type NodeType = 'stochastic' | 'deterministic' | 'constant' | 'observed' | 'plate';

// Define a type for items that can be selected/dragged from the palette
export type PaletteItemType = NodeType | 'add-stochastic-edge' | 'add-deterministic-edge';

// Define a basic node type for the graph
export interface GraphNode {
  id: string;
  name: string;
  type: 'node'; // Discriminator for union type
  nodeType: NodeType; // The specific type of BUGS node
  position: { x: number; y: number; }; // Position on the canvas
  parent?: string; // For compound nodes (plates)

  // BUGS specific properties (optional, and might be combined based on context)
  distribution?: string; // e.g., 'dnorm', 'dbeta' for stochastic nodes
  equation?: string;    // for deterministic nodes, e.g., 'a + b * x'
  observed?: boolean;   // true if this is an observed variable (data)
  initialValue?: any;   // Initial value for MCMC chains (can be a number, string, or object)
  indices?: string;     // For indexed variables, e.g., 'i', 'i,j', '1:N'
  loopVariable?: string; // For plate nodes (e.g., 'i')
  loopRange?: string;    // For plate nodes (e.g., '1:N')
}

// Define a basic edge type for the graph
export interface GraphEdge {
  id: string;
  name?: string; // Optional name for the edge
  type: 'edge'; // Discriminator for union type
  source: string; // ID of the source node
  target: string; // ID of the target node;
  relationshipType?: 'stochastic' | 'deterministic'; // e.g., '~' or '<-'
}

// Union type for all graph elements
export type GraphElement = GraphNode | GraphEdge;

// --- Augment Cytoscape.js types for custom data properties and extensions ---
declare module 'cytoscape' {
  interface NodeSingular {
    // Overload for getting all data
    data(): GraphNode;
    // Overloads for getting specific data properties
    data(key: 'id'): string;
    data(key: 'name'): string;
    data(key: 'type'): 'node';
    data(key: 'nodeType'): NodeType;
    data(key: 'position'): { x: number; y: number; };
    data(key: 'parent'): string | undefined;
    data(key: 'distribution'): string | undefined;
    data(key: 'equation'): string | undefined;
    data(key: 'observed'): boolean | undefined;
    data(key: 'initialValue'): any | undefined;
    data(key: 'indices'): string | undefined;
    data(key: 'loopVariable'): string | undefined;
    data(key: 'loopRange'): string | undefined;
    data(key: string): any;
    // Overloads for setting data properties
    data(key: string, value: any): NodeSingular;
    data(obj: Partial<GraphNode>): NodeSingular;
  }

  interface EdgeSingular {
    // Overload for getting all data
    data(): GraphEdge;
    // Overloads for getting specific data properties
    data(key: 'id'): string;
    data(key: 'name'): string | undefined;
    data(key: 'type'): 'edge';
    data(key: 'source'): string;
    data(key: 'target'): string;
    data(key: 'relationshipType'): 'stochastic' | 'deterministic' | undefined;
    data(key: string): any;
    // Overloads for setting data properties
    data(key: string, value: any): EdgeSingular;
    data(obj: Partial<GraphEdge>): EdgeSingular;
  }

  // Add type definition for the panzoom extension
  interface Core {
    panzoom(options?: any): any;
  }
}
