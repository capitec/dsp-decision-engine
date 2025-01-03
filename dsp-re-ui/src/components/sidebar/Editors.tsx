import React, { useMemo, useCallback, useRef, useState } from "react";
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
} from "../editor/types";
import { useFeatures } from "@components/editor/EditorContext";
import { useNodes } from "@xyflow/react";

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
                  {op}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="w-1/2">
          <Label>Threshold</Label>
          <Input
            type="number"
            step="any"
            value={node.data.threshold}
            onChange={(e) =>
              updateNode({ threshold: parseFloat(e.target.value) })
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

export function LeafConfig({
  node,
  updateNode
}: ConfigProps<LeafNodeData>) {
  const nodes = useNodes<Node>();
  const numLeafNodes = useMemo(() => 
    nodes.reduce(
      (acc: number, n: Node) => n.data.node_type === "leaf" ? acc + 1 : acc,
      0
    ),
    [nodes]
  );

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>Leaf Value (0-{Math.max(0, numLeafNodes - 1)})</Label>
        <Input
          type="number"
          min={0}
          max={Math.max(0, numLeafNodes - 1)}
          value={node.data.leaf_value}
          onChange={(e) => {
            const value = parseInt(e.target.value);
            if (!isNaN(value) && value >= 0 && value < numLeafNodes) {
              updateNode({ leaf_value: value });
            }
          }}
        />
      </div>
    </div>
  );
}
