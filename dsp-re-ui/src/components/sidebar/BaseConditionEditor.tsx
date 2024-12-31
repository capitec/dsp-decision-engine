import React, { useMemo, useState } from 'react';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { 
  Popover, 
  PopoverContent, 
  PopoverTrigger 
} from "@/components/ui/popover";
import { 
  Command, 
  CommandEmpty, 
  CommandGroup, 
  CommandInput, 
  CommandItem, 
  CommandList 
} from "@/components/ui/command";
import { 
  Checkbox 
} from "@/components/ui/checkbox";
import { 
  Input 
} from "@/components/ui/input";
import { 
  Label 
} from "@/components/ui/label";
import { 
  Button 
} from "@/components/ui/button";
import { 
  SidebarGroup, 
  SidebarGroupLabel, 
  SidebarGroupContent 
} from "@/components/ui/sidebar";
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils";
import { EditorState, FlattenedNode } from '@ctx/editor/editorTypes';
import { updateNode } from '@ctx/editor/editorActions';
import { useEditorDispatch } from '@ctx/editor/EditorContext';
import ListManager from './lititem';

interface BaseConditionEditorProps {
  node: FlattenedNode;
  nodeId: string
  editorState: EditorState;
}

export const BaseConditionEditor: React.FC<BaseConditionEditorProps> = ({ 
  node, 
  nodeId,
  editorState
}) => {
  const conditionType = node.condition_type
  const conditionName = node.condition
  const [childrenCount, setChildrenCount] = useState(1);
  const [hasDefaultValue, setHasDefaultValue] = useState(false);
  const [open, setOpen] = useState(false);

  const editorDispatch = useEditorDispatch();

  const existingConditions = useMemo(()=>Array.from(new Set(Object.values(editorState.nodes).reduce(
    (acc, v)=>"condition" in v ? [...acc,v.condition] : acc, [] as string[]
  ))), 
    [editorState.nodes])

  const handleConditionTypeChange = (value: FlattenedNode["condition_type"]) => {
    editorDispatch(updateNode(nodeId, {
      condition_type: value
    }));
    console.log('Changing condition type to:', value);
  };

  const handleConditionNameChange = (value: string) => {
    editorDispatch(updateNode(nodeId, {
      condition: value
    }));
    setOpen(false);
  };

  const handleChildrenCountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const count = parseInt(e.target.value, 10);
    console.log('Changing children count to:', count);
    setChildrenCount(count);
  };

  const handleDefaultValueToggle = (checked: boolean) => {
    console.log('Toggling default value:', checked);
    setHasDefaultValue(checked);
  };

  const children = useMemo(
    ()=>node.connections[0].values.map((v)=>({
      "id": v, 
      //@ts-ignore
      "name": editorState.nodes[v].condition || value 
    })), 
    [editorState.nodes, node]
  )

  return (
    <SidebarGroup>
      <SidebarGroupLabel>Condition Configuration</SidebarGroupLabel>
      <SidebarGroupContent className="space-y-4 p-4">
        {/* Condition Type Dropdown */}
        <div className="space-y-2">
          <Label>Condition Type</Label>
          <Select 
            value={conditionType} 
            onValueChange={handleConditionTypeChange}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select condition type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="base">Base Condition</SelectItem>
              <SelectItem value="table">Expression Condition</SelectItem>
              <SelectItem value="value">Value Node</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Condition Name Autocomplete */}
        <div className="space-y-2">
          <Label>Condition Name</Label>
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                role="combobox"
                aria-expanded={open}
                className="w-full justify-between"
              >
                {conditionName || "Select condition name..."}
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[300px] p-0">
              <Command>
                <CommandInput placeholder="Search conditions..." />
                <CommandList>
                  <CommandEmpty>No conditions found.</CommandEmpty>
                  <CommandGroup>
                    {existingConditions.map((condition) => (
                      <CommandItem
                        key={condition}
                        value={condition}
                        onSelect={handleConditionNameChange}
                      >
                        <Check
                          className={cn(
                            "mr-2 h-4 w-4",
                            conditionName === condition ? "opacity-100" : "opacity-0"
                          )}
                        />
                        {condition}
                      </CommandItem>
                    ))}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
        </div>

        {/* Children Count Input */}
        <div className="space-y-2">
          <Label>Number of Children</Label>
          <Input 
            type="number"
            min={1}
            value={childrenCount}
            onChange={handleChildrenCountChange}
            placeholder="Number of children"
          />
        </div>

        {/* Default Value Checkbox */}
        <div className="flex items-center space-x-2">
          <Checkbox 
            id="default-value"
            checked={hasDefaultValue}
            onCheckedChange={handleDefaultValueToggle}
          />
          <Label htmlFor="default-value">
            Add Default Value Node
          </Label>
        </div>


        <ListManager
          items={children}
          handleAdd={()=>{}}
        />
      </SidebarGroupContent>
    </SidebarGroup>
  );
};

export default BaseConditionEditor;