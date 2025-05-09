import React, { useMemo, useCallback, useRef, useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Command,
  CommandEmpty,
  CommandList,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command";
import { Command as CommandPrimitive } from "cmdk";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import type {
  Node,
  NodeData,
  TreeNode,
  NumericalNodeData,
  CategoricalNodeData,
  LeafNodeData,
  RangeNodeData,
} from "../editor/types";
import { useFeatures, useLeafOrder, useTreeOutput } from "@components/editor/EditorContext";
import { useNodes } from "@xyflow/react";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import { OP_SYMBOL_MAP } from "../editor/util"


interface FeatureEditorProps {
  selectedFeatureId: number;
  onFeatureSelect: (index: number, features: string[]) => void;
  label?: string;
}

interface ConfigProps<T = NodeData> {
  node: Node<T>;
  updateNode: (val: Partial<T>, features?: string[]) => void;
}

export function FeatureEditor({
  selectedFeatureId,
  onFeatureSelect,
  label = "Feature",
}: FeatureEditorProps) {
  const [features, setFeatures] = useFeatures();
  const [searchValue, setSearchValue] = useState("");
  const [popoverOpen, setPopoverOpen] = useState(false);

  const showCreate = useMemo(()=>
    searchValue && (features.findIndex(ft=>ft===searchValue) === -1)
  , [searchValue, features])

  const selectFeature = useCallback((ft: string) => {
    if (!ft) { return }
    setFeatures(fts => {
      let newFts = fts;
      let featureIndex = fts.indexOf(ft);
      if (featureIndex === -1){
        featureIndex = features.length;
        newFts = [...newFts, ft];
      }
      onFeatureSelect(featureIndex, newFts);
      return newFts;
    })
    setPopoverOpen((state) => false)
  },[onFeatureSelect])
  const selectedFeature = useMemo(()=>{
    console.log({selectedFeatureId, features})
    return features[selectedFeatureId]
  }
  , [selectedFeatureId, features])
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" className="w-full justify-start">
            {selectedFeature || "Select feature..."}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="p-0">
          <Command  className="rounded-lg border shadow-md">
            <CommandInput 
              placeholder="Search features..." 
              value={searchValue} 
              onValueChange={(nv) => setSearchValue(nv)}
            />
             <CommandList>
              
             <CommandGroup forceMount>
              {showCreate && 
              <CommandItem 
                key={-1} 
                value={searchValue}
                onSelect={selectFeature}
                forceMount
              >
                {searchValue} (Create New)
              </CommandItem>}
              </CommandGroup>
              <CommandGroup>

              {(features||[]).map((feature, index) => (
                <CommandItem
                  value={feature}
                  onSelect={selectFeature}
                  key={index}
                >
                  {feature}
                </CommandItem>
              ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}



export function NumericalConfig({
  node,
  updateNode,
}: ConfigProps<NumericalNodeData>) {
  const [threshold, setThreshold] = useState(`${node.data.threshold}`);
  useEffect(()=>{
    setThreshold(`${node.data.threshold}`)
  },[node.data.threshold])
  const handleOnBlur = useCallback(()=>{

    const parsedThresh = parseFloat(threshold);
    if (isNaN(parsedThresh)){
      setThreshold(`${node.data.threshold}`)
      return
    }
    updateNode({ threshold: parsedThresh })
  }, [updateNode, threshold, node.data.threshold])


  return (
    <div className="space-y-4">
      <FeatureEditor
        selectedFeatureId={node.data.split_feature_id}
        onFeatureSelect={(index, features) => updateNode({ split_feature_id: index }, features)}
      />
      <div className="flex items-center space-x-4">
        <div className="w-1/2">
          <Label>Comparison</Label>
          <Select
            value={node.data.comparison_op}
            onValueChange={(value: NumericalNodeData["comparison_op"]) => updateNode({ comparison_op: value })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["<=", "<", "==", ">", ">="].map((op) => (
                <SelectItem key={op} value={op}>
                  {OP_SYMBOL_MAP[op]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="w-1/2">
          <Label>Threshold</Label>
          <Input
            type="text"
            value={threshold}
            placeholder="Enter a number"
            onBlur={handleOnBlur}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleOnBlur();
              }
            }}
            onChange={(e) =>
              setThreshold(e.target.value)
            }
          />
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox
          id="default-left"
          checked={node.data.default_left}
          onCheckedChange={(checked) => updateNode({ default_left: !!checked })}
        />
        <Label htmlFor="default-left">Default Left</Label>
      </div>
    </div>
  );
}

export function CategoricalConfig({
  node,
  updateNode
}: ConfigProps<CategoricalNodeData>) {

  const addCategory = (value: string) => {
    const newCategory = parseInt(value);
    if (!isNaN(newCategory) && !node.data.category_list.includes(newCategory)) {
      updateNode({ 
        category_list: [...node.data.category_list, newCategory].sort((a, b) => a - b) 
      });
    }
  };

  const removeCategory = (category: number) => {
    updateNode({
      category_list: node.data.category_list.filter(c => c !== category)
    });
  };

  return (
    <div className="space-y-4">
      <FeatureEditor
        selectedFeatureId={node.data.split_feature_id}
        onFeatureSelect={(index, features) => updateNode({ split_feature_id: index }, features)}
      />

      <div className="space-y-2">
        <Label>Categories</Label>
        <div className="flex flex-wrap gap-2">
          {node.data.category_list.map((category) => (
            <Badge
              key={category}
              variant="secondary"
              className="flex items-center gap-1"
            >
              {category}
              <button
                className="ml-1 hover:bg-red-500 rounded-full"
                onClick={() => removeCategory(category)}
              >
                ×
              </button>
            </Badge>
          ))}
        </div>
        <Input
          type="number"
          placeholder="Add category..."
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              //@ts-ignore
              addCategory(e.target.value);
              //@ts-ignore
              e.target.value = "";
            }
          }}
        />
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox
          id="default-left"
          checked={node.data.default_left}
          onCheckedChange={(checked) => updateNode({ default_left: !!checked })}
        />
        <Label htmlFor="default-left">Default Left</Label>
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox
          id="category-right"
          checked={node.data.category_list_right_child}
          onCheckedChange={(checked) =>
            updateNode({ category_list_right_child: !!checked })
          }
        />
        <Label htmlFor="category-right">Categories go to right child</Label>
      </div>
    </div>
  );
}


export function RangeConfigOld({
  node,
  updateNode,
}: ConfigProps<RangeNodeData>) {
  const [newThreshold, setNewThreshold] = useState("");
  const [editIndex, setEditIndex] = useState<number | null>(null);
  const [editValue, setEditValue] = useState("");

  // Sort thresholds when updating
  const updateThresholds = (thresholds: number[]) => {
    updateNode({ 
      thresholds: [...thresholds].sort((a, b) => a - b) 
    });
  };

  // Add a new threshold
  const addThreshold = () => {
    const value = parseFloat(newThreshold);
    const thresholds = node.data.thresholds || [];
    if (!isNaN(value) && !thresholds.includes(value)) {
      updateThresholds([...thresholds, value]);
      setNewThreshold("");
    }
  };

  // Remove a threshold
  const removeThreshold = (threshold: number) => {
    const thresholds = node.data.thresholds || [];
    updateThresholds(thresholds.filter(t => t !== threshold));
  };

  // Start editing a threshold
  const startEdit = (index: number) => {
    setEditIndex(index);
    setEditValue(`${node.data.thresholds[index]}`);
  };

  // Save edited threshold
  const saveEdit = () => {
    if (editIndex !== null) {
      const value = parseFloat(editValue);
      if (!isNaN(value)) {
        const newThresholds = [...node.data.thresholds];
        newThresholds[editIndex] = value;
        updateThresholds(newThresholds);
      }
      setEditIndex(null);
    }
  };

  // Cancel editing
  const cancelEdit = () => {
    setEditIndex(null);
  };

  // Handle keyboard events for editing
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      saveEdit();
    } else if (e.key === "Escape") {
      cancelEdit();
    }
  };

  return (
    <div className="space-y-4">
      <FeatureEditor
        selectedFeatureId={node.data.split_feature_id}
        onFeatureSelect={(index, features) => updateNode({ split_feature_id: index }, features)}
      />

      <div className="space-y-2">
        <Label>Thresholds</Label>
        
        {/* Display thresholds as ranges */}
        <div className="space-y-2">
          {node.data.thresholds && (
            <div className="text-sm text-muted-foreground italic">
              No thresholds defined yet. Add thresholds to create ranges.
            </div>
          )}
          
          {/* First range (if any thresholds exist) */}
          {node.data.thresholds && (
            <div className="flex items-center space-x-2 p-2 bg-muted/50 rounded-md">
              <span className="text-sm">feature &lt; </span>
              {editIndex === 0 ? (
                <Input
                  type="text"
                  className="w-24"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onBlur={saveEdit}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
              ) : (
                <button
                  className="px-2 py-1 bg-muted rounded hover:bg-muted/80"
                  onClick={() => startEdit(0)}
                >
                  {node.data.thresholds[0]}
                </button>
              )}
              <button
                className="ml-2 hover:text-destructive"
                onClick={() => removeThreshold(node.data.thresholds[0])}
              >
                <span className="sr-only">Remove</span>
                ×
              </button>
            </div>
          )}
          
          {/* Middle ranges */}
          {(node.data.thresholds || []).slice(0, -1).map((threshold, index) => (
            <div key={threshold} className="flex items-center space-x-2 p-2 bg-muted/50 rounded-md">
              <span className="text-sm">feature ≥ </span>
              {editIndex === index ? (
                <Input
                  type="text"
                  className="w-24"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onBlur={saveEdit}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
              ) : (
                <button
                  className="px-2 py-1 bg-muted rounded hover:bg-muted/80"
                  onClick={() => startEdit(index)}
                >
                  {threshold}
                </button>
              )}
              <span className="text-sm">&amp; feature &lt; </span>
              {editIndex === index + 1 ? (
                <Input
                  type="text"
                  className="w-24"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onBlur={saveEdit}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
              ) : (
                <button
                  className="px-2 py-1 bg-muted rounded hover:bg-muted/80"
                  onClick={() => startEdit(index + 1)}
                >
                  {node.data.thresholds[index + 1]}
                </button>
              )}
            </div>
          ))}
          
          {/* Last range (if multiple thresholds exist) */}
          {node.data.thresholds && node.data.thresholds.length > 1 && (
            <div className="flex items-center space-x-2 p-2 bg-muted/50 rounded-md">
              <span className="text-sm">feature ≥ </span>
              {editIndex === node.data.thresholds.length - 1 ? (
                <Input
                  type="text"
                  className="w-24"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onBlur={saveEdit}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
              ) : (
                <button
                  className="px-2 py-1 bg-muted rounded hover:bg-muted/80"
                  onClick={() => startEdit(node.data.thresholds.length - 1)}
                >
                  {node.data.thresholds[node.data.thresholds.length - 1]}
                </button>
              )}
              <button
                className="ml-2 hover:text-destructive"
                onClick={() => removeThreshold(node.data.thresholds[node.data.thresholds.length - 1])}
              >
                <span className="sr-only">Remove</span>
                ×
              </button>
            </div>
          )}
        </div>

        {/* Add new threshold */}
        <div className="flex space-x-2 mt-2">
          <Input
            type="text"
            placeholder="Add threshold..."
            value={newThreshold}
            onChange={(e) => setNewThreshold(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                addThreshold();
              }
            }}
          />
          <Button 
            type="button"
            variant="secondary" 
            size="sm" 
            onClick={addThreshold}
            disabled={!newThreshold || isNaN(parseFloat(newThreshold))}
          >
            Add
          </Button>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox
          id="default-left"
          checked={node.data.default_left}
          onCheckedChange={(checked) => updateNode({ default_left: !!checked })}
        />
        <Label htmlFor="default-left">Default Left</Label>
      </div>
    </div>
  );
}

// ThresholdInput component for better number editing
const ThresholdInput = ({
  value,
  onUpdate,
  className = "w-24",
}: {
  value: number;
  onUpdate: (value: number) => void;
  className?: string;
}) => {
  const [editValue, setEditValue] = useState(`${value}`);
  
  // Update edit value when the prop value changes
  useEffect(() => {
    setEditValue(`${value}`);
  }, [value]);
  
  // Handle blur event to validate and update
  const handleOnBlur = useCallback(() => {
    const parsedValue = parseFloat(editValue);
    
    // If invalid, reset to original value
    if (isNaN(parsedValue)) {
      setEditValue(`${value}`);
      return;
    }
    
    // Update with new value
    onUpdate(parsedValue);
  }, [editValue, value, onUpdate]);
  
  // Format for display
  const formatValue = (val: string) => {
    return val.length > 8 ? val.substring(0, 7) + '...' : val;
  };
  
  return (
    <Input
      type="text"
      className={className}
      value={editValue}
      onChange={(e) => setEditValue(e.target.value)}
      onBlur={handleOnBlur}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          handleOnBlur();
        }
      }}
    />
  );
};

export function RangeConfig({
  node,
  updateNode,
}: ConfigProps<RangeNodeData>) {
  // Ensure thresholds array exists
  const thresholds = node.data.thresholds || [];
  const [newThreshold, setNewThreshold] = useState("");
  
  // Initialize data if missing
  useEffect(() => {
    if (!node.data.thresholds) {
      updateNode({ thresholds: [] });
    }
  }, [node.data, updateNode]);

  // Sort thresholds when updating
  const updateThresholds = (newThresholds: number[]) => {
    const sorted = [...newThresholds].sort((a, b) => a - b);
    updateNode({ thresholds: sorted });
  };

  // Update a threshold at specific index
  const updateThresholdAt = (index: number, value: number) => {
    // Check if value already exists elsewhere in the thresholds
    const otherThresholds = thresholds.filter((_, i) => i !== index);
    if (otherThresholds.includes(value)) {
      // If duplicate, don't update
      return;
    }
    
    // Create new thresholds array with updated value
    const newThresholds = [...thresholds];
    newThresholds[index] = value;
    updateThresholds(newThresholds);
  };

  // Add a new threshold
  const addThreshold = () => {
    const value = parseFloat(newThreshold);
    if (!isNaN(value) && !thresholds.includes(value)) {
      updateThresholds([...thresholds, value]);
      setNewThreshold("");
    }
  };

  // Remove a threshold by index
  const removeThreshold = (index: number) => {
    const newThresholds = [...thresholds];
    newThresholds.splice(index, 1);
    updateThresholds(newThresholds);
  };

  return (
    <div className="space-y-4">
      <FeatureEditor
        selectedFeatureId={node.data.split_feature_id}
        onFeatureSelect={(index, features) => updateNode({ split_feature_id: index }, features)}
      />

      <div className="space-y-2">
        <Label>Thresholds</Label>
        
        {/* Display thresholds as ranges */}
        <div className="space-y-2">
          {thresholds.length === 0 && (
            <div className="text-sm text-muted-foreground italic">
              No thresholds defined yet. Add thresholds to create ranges.
            </div>
          )}
          
          {/* First range (if any thresholds exist) */}
          {thresholds.length > 0 && (
            <div className="flex items-center space-x-2 p-2 bg-muted/50 rounded-md">
              <span className="text-sm">&lt; </span>
              <ThresholdInput
                value={thresholds[0]}
                onUpdate={(value) => updateThresholdAt(0, value)}
              />
              <div className="flex-grow"></div>
              <button
                className="text-lg hover:text-destructive"
                onClick={() => removeThreshold(0)}
                title="Remove"
              >
                ×
              </button>
            </div>
          )}
          
          {/* Middle ranges */}
          {thresholds.slice(0, -1).map((threshold, index) => (
            <div key={`${threshold}-${index}`} className="flex items-center space-x-2 p-2 bg-muted/50 rounded-md">
              <span className="text-sm">≥ </span>
              <ThresholdInput
                value={threshold}
                onUpdate={(value) => updateThresholdAt(index, value)}
              />
              <span className="text-sm">&lt; </span>
              <span className="px-2 py-1 bg-muted/80 rounded w-24 text-center overflow-hidden">
                {thresholds[index + 1].toString().length > 8 
                  ? thresholds[index + 1].toString().substring(0, 7) + '...' 
                  : thresholds[index + 1]}
              </span>
              <div className="flex-grow"></div>
              <button
                className="text-lg hover:text-destructive"
                onClick={() => removeThreshold(index)}
                title="Remove"
              >
                ×
              </button>
            </div>
          ))}
          
          {/* Last range (if multiple thresholds exist) */}
          {thresholds.length > 1 && (
            <div className="flex items-center space-x-2 p-2 bg-muted/50 rounded-md">
              <span className="text-sm">≥ </span>
              <ThresholdInput
                value={thresholds[thresholds.length - 1]}
                onUpdate={(value) => updateThresholdAt(thresholds.length - 1, value)}
              />
              <div className="flex-grow"></div>
              <button
                className="text-lg hover:text-destructive"
                onClick={() => removeThreshold(thresholds.length - 1)}
                title="Remove"
              >
                ×
              </button>
            </div>
          )}
        </div>

        {/* Add new threshold */}
        <div className="flex space-x-2 mt-2">
          <Input
            type="text"
            placeholder="Add threshold..."
            value={newThreshold}
            onChange={(e) => setNewThreshold(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                addThreshold();
              }
            }}
          />
          <Button 
            type="button"
            variant="secondary" 
            size="sm" 
            onClick={addThreshold}
            disabled={!newThreshold || isNaN(parseFloat(newThreshold))}
          >
            Add
          </Button>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox
          id="default-left"
          checked={node.data.default_left !== undefined ? node.data.default_left : false}
          onCheckedChange={(checked) => updateNode({ default_left: !!checked })}
        />
        <Label htmlFor="default-left">Default Left</Label>
      </div>
    </div>
  );
}
interface TreeOutput {
  data: string[][]
  columns: string[]
}
export function LeafConfig({
  node,
  updateNode
}: ConfigProps<LeafNodeData>) {
  const nodes = useNodes<Node>();

  const [outputs, setOutputs] = useTreeOutput();
  const [leafOrder, updateLeafOrder] = useLeafOrder();
  const leafIndex = useMemo(() => leafOrder.indexOf(node.id),
    [leafOrder, node]
  );
  const numLeafNodes = useMemo(() => leafOrder.length,
    [leafOrder]
  );
  const selectedRow = outputs.data[node.data.leaf_value+1];

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>{outputs.data.length?`Leaf Output Value (-1 – ${outputs.data.length-2})`: "Leaf Output Value"}</Label>
        {
          outputs.data.length?(
            <Input 
              type="number"
              min={-1}
              max={outputs.data.length-2}
              value={node.data.leaf_value}
              onChange={(e) => {
                const value = parseInt(e.target.value);
                if (!isNaN(value) && value >= -1 && value < outputs.data.length) {
                  updateNode({ leaf_value: value });
                }
              }}
            />
          ):(
            <Input 
              type="text"
              disabled
              value="Please configure outputs in the output editor"
            />
          )
        }
        <ScrollArea className="h-24 border rounded-md p-2 overflow-auto">
          <div className="space-y-1">
            {outputs.columns.map((column, i) => (
              <div key={i} className="flex text-sm">
                <span className="font-medium w-24 truncate">{column}:</span>
                <span className="text-muted-foreground truncate">{selectedRow?.[i]}</span>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>
      <div className="space-y-2">
        <Label>Leaf Order (0-{numLeafNodes-1})</Label>
        <Input
          type="number"
          min={0}
          max={numLeafNodes-1}
          value={leafIndex}
          onChange={(e) => {
            const value = parseInt(e.target.value);
            if (!isNaN(value) && value >= 0 && value < numLeafNodes) {
              updateLeafOrder(node.id, value);
            }
          }}
        />
      </div>
    </div>
  );
}
