// import { NodeEditor, GetSchemes, ClassicPreset } from 'rete';

// type Schemes = GetSchemes<ClassicPreset.Node, ClassicPreset.Connection>;

// interface WebSocketPluginOptions {
//   url: string;
//   reconnectDelay?: number;
//   onConnect?: () => void;
//   onDisconnect?: () => void;
// }

// interface WebSocketMessage {
//   type: 'nodeAdd' | 'nodeRemove' | 'connectionAdd' | 'connectionRemove' | 'nodeUpdate';
//   payload: any;
// }

// export class WebSocketPlugin {
//   private editor!: NodeEditor<Schemes>;
//   private ws!: WebSocket;
//   private options: WebSocketPluginOptions;
//   private reconnectTimeout: NodeJS.Timeout | null = null;
//   private isConnected = false;

//   constructor(options: WebSocketPluginOptions) {
//     this.options = {
//       reconnectDelay: 3000,
//       ...options
//     };
//   }

//   async install(editor: NodeEditor<Schemes>) {
//     this.editor = editor;
//     await this.setupWebSocket();
//     this.setupEditorListeners();
//   }

//   private async setupWebSocket() {  
//     const a = new ClassicPreset.Node("Override styles");
//     a.addOutput("a", new ClassicPreset.Output(socket));
//     a.addInput("a", new ClassicPreset.Input(socket));
//     await editor.addNode(a);
  
//     const b = new ClassicPreset.Node("Fully customized");
//     b.addOutput("a", new ClassicPreset.Output(socket));
//     b.addInput("a", new ClassicPreset.Input(socket));
//     await editor.addNode(b);
  
//     await area.translate(a.id, { x: 0, y: 0 });
//     await area.translate(b.id, { x: 300, y: 0 });
  
//     await editor.addConnection(new ClassicPreset.Connection(a, "a", b, "a"));

//   }
//   private setupWebSocketReal() {
//     this.ws = new WebSocket(this.options.url);
    
//     this.ws.onopen = () => {
//       this.isConnected = true;
//       this.options.onConnect?.();
      
//       // Request initial state
//       this.send({
//         type: 'requestState',
//         payload: {}
//       });
//     };

//     this.ws.onclose = () => {
//       this.isConnected = false;
//       this.options.onDisconnect?.();
//       this.scheduleReconnect();
//     };

//     this.ws.onmessage = (event) => {
//       const message: WebSocketMessage = JSON.parse(event.data);
//       this.handleIncomingMessage(message);
//     };
//   }

//   private setupEditorListeners() {
//     // // @ts-ignore
//     // editor.addPipe(context => {
//     //     console.log({context})
//     // })
//     // // @ts-ignore
//     // area.addPipe(context => {
//     //     if (context.type === "nodetranslated" || context.type === "nodepicked"){
//     //     console.log({context})
//     //     }
//     //     return context
//     // })
//     // Listen for local changes
//     // this.editor.on('nodecreate', node => {
//     //   if (this.isConnected) {
//     //     this.send({
//     //       type: 'nodeAdd',
//     //       payload: node
//     //     });
//     //   }
//     // });

//     // this.editor.on('noderemove', node => {
//     //   if (this.isConnected) {
//     //     this.send({
//     //       type: 'nodeRemove',
//     //       payload: { id: node.id }
//     //     });
//     //   }
//     // });

//     // this.editor.on('connectioncreate', connection => {
//     //   if (this.isConnected) {
//     //     this.send({
//     //       type: 'connectionAdd',
//     //       payload: connection
//     //     });
//     //   }
//     // });

//     // this.editor.on('connectionremove', connection => {
//     //   if (this.isConnected) {
//     //     this.send({
//     //       type: 'connectionRemove',
//     //       payload: connection
//     //     });
//     //   }
//     // });
//   }

//   private async handleIncomingMessage(message: WebSocketMessage) {
//     try {
//       switch (message.type) {
//         case 'nodeAdd':
//           await this.editor.addNode(message.payload);
//           break;
//         case 'nodeRemove':
//           const node = this.editor.getNode(message.payload.id);
//           if (node) await this.editor.removeNode(node);
//           break;
//         case 'connectionAdd':
//           await this.editor.addConnection(message.payload);
//           break;
//         case 'connectionRemove':
//           await this.editor.removeConnection(message.payload.id);
//           break;
//         case 'nodeUpdate':
//           const existingNode = this.editor.getNode(message.payload.id);
//           if (existingNode) {
//             Object.assign(existingNode, message.payload.data);
//             await this.editor.updateNode(existingNode);
//           }
//           break;
//       }
//     } catch (error) {
//       console.error('Error handling WebSocket message:', error);
//     }
//   }

//   private send(message: WebSocketMessage) {
//     if (this.isConnected) {
//       this.ws.send(JSON.stringify(message));
//     }
//   }

//   private scheduleReconnect() {
//     if (this.reconnectTimeout) {
//       clearTimeout(this.reconnectTimeout);
//     }
    
//     this.reconnectTimeout = setTimeout(() => {
//       this.setupWebSocket();
//     }, this.options.reconnectDelay);
//   }

//   destroy() {
//     if (this.reconnectTimeout) {
//       clearTimeout(this.reconnectTimeout);
//     }
//     if (this.ws) {
//       this.ws.close();
//     }
//   }
// }