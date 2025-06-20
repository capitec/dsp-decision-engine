import React, { useState } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import { Button } from '@/components/ui/button'; // Assuming you're using ShadCN UI library

// Define the type for each item
type Item = {
  id: string;
  name: string;
};

// Define the props for the ListItem component
interface ListItemProps {
  item: Item;
  index: number;
  onDelete: (id: string) => void;
}

// ListItem Component
const ListItem: React.FC<ListItemProps> = ({ item, index, onDelete }) => (
  <Draggable key={item.id} draggableId={item.id} index={index}>
    {(provided) => (
      <div
        ref={provided.innerRef}
        {...provided.draggableProps}
        {...provided.dragHandleProps}
        className="flex items-center p-2 border-b"
        style={provided.draggableProps.style}
      >

        {/* Item name */}
        <div className="flex-grow">{item.name}</div>

        {/* Delete button */}
        <button
          onClick={() => onDelete(item.id)}
          className="text-red-500 hover:text-red-700"
        >
          🗑️
        </button>
      </div>
    )}
  </Draggable>
);

interface OrderedItem {
    id: string, name: string
}
// Define the props for the ListManager component
interface ListManagerProps {
    items: OrderedItem[],
    handleAdd: ()=>void,
    handleDelete: (id: string)=>void,
    handleReorder: (reorderedItems:OrderedItem[])=>void,
}

// ListManager Component
const ListManager: React.FC<ListManagerProps> = ({items, handleAdd, handleDelete, handleReorder}) => {
//   const [items, setItems] = useState<Item[]>([
//     { id: '1', name: 'Item 1' },
//     { id: '2', name: 'Item 2' },
//     { id: '3', name: 'Item 3' },
//   ]);

//   // Handle item deletion
//   const handleDelete = (id: string): void => {
//     setItems((prevItems) => prevItems.filter(item => item.id !== id));
//   };

//   // Handle item addition
//   const handleAdd = (): void => {
//     const newItem: Item = { id: String(items.length + 1), name: `Item ${items.length + 1}` };
//     setItems((prevItems) => [...prevItems, newItem]);
//   };

  const handleOnDragEnd = (result: DropResult): void => {
    const { source, destination } = result;

    if (!destination || source.index === destination.index) {
      return;
    }

    const reorderedItems = Array.from(items);
    const [removed] = reorderedItems.splice(source.index, 1);
    reorderedItems.splice(destination.index, 0, removed);

    handleReorder(reorderedItems);
  };

  return (
    <div className="p-4">
      <DragDropContext onDragEnd={handleOnDragEnd}>
        <Droppable droppableId="droppable">
          {(provided) => (
            <div
              ref={provided.innerRef}
              {...provided.droppableProps}
              className="space-y-2"
            >
              {items.map((item, index) => (
                <ListItem key={item.id} item={item} index={index} onDelete={handleDelete} />
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>

      <Button className="mt-4" onClick={handleAdd}>
        Add New Item
      </Button>
    </div>
  );
};

export default ListManager;
