"""Data models for RAPTOR tree structure."""

import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TreeNode:
    """A node in the RAPTOR tree."""
    
    # Content
    text: str
    
    # Tree structure
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: int = 0  # 0 = leaf (original chunks), increases upward
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding (computed lazily)
    _embedding: Optional[List[float]] = field(default=None, repr=False)
    
    # Source tracking
    source_doc: str = ""  # Which document this came from
    page: int = 0
    chunk_id: Optional[int] = None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (original chunk)."""
        return self.level == 0
    
    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.parent is None
    
    @property
    def embedding(self) -> Optional[List[float]]:
        """Get embedding if computed."""
        return self._embedding
    
    @embedding.setter
    def embedding(self, value: List[float]):
        """Set embedding."""
        self._embedding = value
    
    def get_leaf_nodes(self) -> List['TreeNode']:
        """Get all leaf nodes under this node."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves
    
    def get_path_to_root(self) -> List['TreeNode']:
        """Get path from this node to root."""
        path = [self]
        current = self
        while current.parent:
            current = current.parent
            path.append(current)
        return path
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'text': self.text,
            'level': self.level,
            'parent_id': self.parent.node_id if self.parent else None,
            'children_ids': [c.node_id for c in self.children],
            'metadata': self.metadata,
            'embedding': self._embedding,
            'source_doc': self.source_doc,
            'page': self.page,
            'chunk_id': self.chunk_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict, nodes_by_id: Dict[str, 'TreeNode'] = None) -> 'TreeNode':
        """Create node from dictionary."""
        node = cls(
            text=data['text'],
            node_id=data['node_id'],
            level=data['level'],
            metadata=data.get('metadata', {}),
            source_doc=data.get('source_doc', ''),
            page=data.get('page', 0),
            chunk_id=data.get('chunk_id')
        )
        node._embedding = data.get('embedding')
        return node


@dataclass
class RaptorTree:
    """The complete RAPTOR tree structure."""
    
    # All nodes indexed by ID
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    
    # Root nodes (usually one per document)
    roots: List[TreeNode] = field(default_factory=list)
    
    # Leaf nodes (original chunks)
    leaves: List[TreeNode] = field(default_factory=list)
    
    # Tree configuration
    max_levels: int = 3
    target_chunk_size: int = 100  # Target size for summary chunks
    
    def add_node(self, node: TreeNode, parent: Optional[TreeNode] = None):
        """Add a node to the tree."""
        self.nodes[node.node_id] = node
        
        if parent:
            node.parent = parent
            parent.children.append(node)
        else:
            # This is a root node
            self.roots.append(node)
        
        if node.is_leaf:
            self.leaves.append(node)
    
    def get_nodes_at_level(self, level: int) -> List[TreeNode]:
        """Get all nodes at a specific level."""
        return [n for n in self.nodes.values() if n.level == level]
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all node embeddings as a matrix."""
        embeddings = []
        for node in self.nodes.values():
            if node.embedding:
                embeddings.append(node.embedding)
        return np.array(embeddings) if embeddings else np.array([])
    
    def get_flattened_chunks(self) -> List[TreeNode]:
        """Get all nodes as flat list (for comparison with standard RAG)."""
        return list(self.nodes.values())
    
    def to_dict(self) -> Dict:
        """Serialize tree to dictionary."""
        return {
            'nodes': {nid: n.to_dict() for nid, n in self.nodes.items()},
            'roots': [r.node_id for r in self.roots],
            'leaves': [l.node_id for l in self.leaves],
            'max_levels': self.max_levels,
            'target_chunk_size': self.target_chunk_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RaptorTree':
        """Deserialize tree from dictionary."""
        tree = cls(
            max_levels=data.get('max_levels', 3),
            target_chunk_size=data.get('target_chunk_size', 100)
        )
        
        # Create all nodes first
        for node_id, node_data in data['nodes'].items():
            node = TreeNode.from_dict(node_data)
            tree.nodes[node_id] = node
        
        # Reconnect parent-child relationships
        for node_id, node_data in data['nodes'].items():
            node = tree.nodes[node_id]
            if node_data['parent_id'] and node_data['parent_id'] in tree.nodes:
                node.parent = tree.nodes[node_data['parent_id']]
            for child_id in node_data['children_ids']:
                if child_id in tree.nodes:
                    node.children.append(tree.nodes[child_id])
        
        # Set roots and leaves
        tree.roots = [tree.nodes[r] for r in data['roots'] if r in tree.nodes]
        tree.leaves = [tree.nodes[l] for l in data['leaves'] if l in tree.nodes]
        
        return tree
