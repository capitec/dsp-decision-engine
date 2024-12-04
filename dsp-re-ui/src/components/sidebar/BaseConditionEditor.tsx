import React, { useState } from 'react';
import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarRail,
  } from "@/components/ui/sidebar"
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Trash2, Plus } from 'lucide-react';

interface BaseConditionEditorProps {
  node: {
    condition_type: string;
    condition: string;
    values: string[];
  };
  onUpdate: (updatedNode: any) => void;
}

interface ValueEntry {
  id: string;
  type: 'boolean' | 'table';
  value: string;
}

export const BaseConditionEditor: React.FC<BaseConditionEditorProps> = ({ node, onUpdate }) => {
  const [values, setValues] = useState<ValueEntry[]>(
    node.values.map((value, index) => ({
      id: `value-${index}`,
      type: 'boolean', // Default to boolean, could be enhanced to detect type
      value: value
    }))
  );
  const [defaultValue, setDefaultValue] = useState<string>(JSON.stringify(node.values || {}));

  const addValue = () => {
    console.log("Add value")
    // const newValues = [...values, {
    //   id: `value-${Date.now()}`,
    //   type: 'boolean',
    //   value: ''
    // }];
    // setValues(newValues);
  };

  const removeValue = (idToRemove: string) => {
    const newValues = values.filter(v => v.id !== idToRemove);
    setValues(newValues);
  };

  const updateValue = (id: string, newValue: Partial<ValueEntry>) => {
    const newValues = values.map(v => 
      v.id === id ? { ...v, ...newValue } : v
    );
    setValues(newValues);
  };

  const handleSave = () => {
    try {
      const parsedDefaultValue = JSON.parse(defaultValue);
      const updatedNode = {
        ...node,
        values: values.map(v => v.value),
        default_value: parsedDefaultValue
      };
      onUpdate(updatedNode);
    } catch (error) {
      console.error('Invalid JSON for default value', error);
      // TODO: Add user-facing error handling
    }
  };

  return (

    <SidebarGroup>
    <SidebarGroupLabel>Base Condition Editor: {node.condition}</SidebarGroupLabel>
    <SidebarGroupContent>

        <div className="space-y-4">
          {/* Default Value Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Default Value (JSON)
            </label>
            <Input 
              value={defaultValue}
              onChange={(e) => setDefaultValue(e.target.value)}
              placeholder="Enter default value as JSON"
              className="w-full"
            />
          </div>

          {/* Values Section */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-md font-semibold">Values</h3>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={addValue}
                className="flex items-center"
              >
                <Plus className="mr-2 h-4 w-4" /> Add Value
              </Button>
            </div>

            {values.map((valueEntry) => (
              <div 
                key={valueEntry.id} 
                className="flex items-center space-x-2 mb-2"
              >
                <Select 
                  value={valueEntry.type}
                  onValueChange={(newType) => 
                    updateValue(valueEntry.id, { type: newType as 'boolean' | 'table' })
                  }
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="boolean">Boolean Condition</SelectItem>
                    <SelectItem value="table">Table Condition</SelectItem>
                  </SelectContent>
                </Select>

                <Input 
                  value={valueEntry.value}
                  onChange={(e) => 
                    updateValue(valueEntry.id, { value: e.target.value })
                  }
                  placeholder="Enter value"
                  className="flex-grow"
                />

                <Button 
                  variant="destructive" 
                  size="icon" 
                  onClick={() => removeValue(valueEntry.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>

          {/* Save Button */}
          <Button 
            onClick={handleSave}
            className="w-full"
          >
            Save Changes
          </Button>
        </div>

    </SidebarGroupContent>
    </SidebarGroup>
  );
};

export default BaseConditionEditor;