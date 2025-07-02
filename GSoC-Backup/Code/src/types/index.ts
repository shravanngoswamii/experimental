/**
 * Defines the core data structures for the DoodleBUGS application.
 */

// Represents the fundamental type of a BUGS model node.
export type NodeType = 'stochastic' | 'deterministic' | 'constant' | 'observed' | 'plate';

// Represents any item that can be selected from the left-hand palette.
export type PaletteItemType = NodeType | 'add-edge';

// Interface for a graph node element.
export interface GraphNode {
  id: string;
  name: string;
  type: 'node';
  nodeType: NodeType;
  position: { x: number; y: number; };
  parent?: string; // ID of the parent plate, if any.
  distribution?: string;
  equation?: string;
  observed?: boolean;
  initialValue?: any;
  indices?: string;
  loopVariable?: string;
  loopRange?: string;
}

// Interface for a graph edge element.
export interface GraphEdge {
  id: string;
  name?: string;
  type: 'edge';
  source: string;
  target: string;
}

// A union type for any element that can exist on the graph.
export type GraphElement = GraphNode | GraphEdge;

// This interface represents the data loaded from an example's model.json file.
export interface ExampleModel {
  name: string;
  graphJSON: GraphElement[];
}


/**
 * Augments the official Cytoscape.js type definitions to provide
 * better type-safety for data properties specific to this application.
 */
declare module 'cytoscape' {
  // By augmenting the Core interface, we can add types for extensions.
  interface Core {
    panzoom(options?: any): any;
  }

  interface NodeSingular {
    data(key: string): any;
    data(): GraphNode;
  }

  interface EdgeSingular {
    data(): GraphEdge & { relationshipType?: 'stochastic' | 'deterministic' };
  }
}
