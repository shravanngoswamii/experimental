export type NodeType = 'stochastic' | 'deterministic' | 'constant' | 'observed' | 'plate';

export type PaletteItemType = NodeType | 'add-stochastic-edge' | 'add-deterministic-edge';
export interface GraphNode {
  id: string;
  name: string;
  type: 'node';
  nodeType: NodeType;
  position: { x: number; y: number; };
  parent?: string;


  distribution?: string;
  equation?: string;
  observed?: boolean;

  initialValue?: any;
  indices?: string;
  loopVariable?: string;
  loopRange?: string;
}

export interface GraphEdge {
  id: string;
  name?: string;
  type: 'edge';
  source: string;
  target: string;
  relationshipType?: 'stochastic' | 'deterministic';
}

export type GraphElement = GraphNode | GraphEdge;

declare module 'cytoscape' {
  interface NodeSingular {

    data(): GraphNode;

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


    data(key: string, value: any): NodeSingular;
    data(obj: Partial<GraphNode>): NodeSingular;
  }

  interface EdgeSingular {

    data(): GraphEdge;

    data(key: 'id'): string;
    data(key: 'name'): string | undefined;
    data(key: 'type'): 'edge';
    data(key: 'source'): string;
    data(key: 'target'): string;
    data(key: 'relationshipType'): 'stochastic' | 'deterministic' | undefined;
    data(key: string): any;
    data(key: string, value: any): EdgeSingular;
    data(obj: Partial<GraphEdge>): EdgeSingular;
  }

  
  interface Core {

    panzoom(options?: any): any;
  }
}
