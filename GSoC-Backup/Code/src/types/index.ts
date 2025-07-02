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

/**
 * Augments the official Cytoscape.js type definitions to provide
 * better type-safety for data properties specific to this application.
 */
declare module 'cytoscape' {
  // By augmenting the Core interface, we can add types for extensions.
  interface Core {
    // FIX: Add the panzoom extension's type definition.
    panzoom(options?: any): any;
  }

  interface NodeSingular {
    // FIX: Add a generic string key signature to the data method.
    // This allows calling .data('name'), .data('id'), etc., without TypeScript errors.
    data(key: string): any;
    data(): GraphNode;
  }

  interface EdgeSingular {
    // Overload data() to include our dynamic relationshipType property.
    data(): GraphEdge & { relationshipType?: 'stochastic' | 'deterministic' };
  }
}
