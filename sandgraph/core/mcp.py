"""
MCP (Model Communication Protocol) 协议定义

用于LLM与沙盒之间的标准化通信
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import uuid
from datetime import datetime


class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"


class ActionType(Enum):
    """动作类型枚举"""
    CASE_GENERATOR = "case_generator"
    PROMPT_FUNC = "prompt_func"
    VERIFY_SCORE = "verify_score"
    FULL_CYCLE = "full_cycle"


@dataclass
class MCPMessage:
    """MCP消息基类"""
    message_id: str
    message_type: MessageType
    timestamp: datetime
    sender: str
    receiver: str
    data: Dict[str, Any]
    
    def __init__(self, message_type: MessageType, sender: str, receiver: str, data: Dict[str, Any]):
        self.message_id = str(uuid.uuid4())
        self.message_type = message_type
        self.timestamp = datetime.now()
        self.sender = sender
        self.receiver = receiver
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "receiver": self.receiver,
            "data": self.data
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """从字典创建消息"""
        msg = cls.__new__(cls)
        msg.message_id = data["message_id"]
        msg.message_type = MessageType(data["message_type"])
        msg.timestamp = datetime.fromisoformat(data["timestamp"])
        msg.sender = data["sender"]
        msg.receiver = data["receiver"]
        msg.data = data["data"]
        return msg


@dataclass
class MCPRequest(MCPMessage):
    """MCP请求消息"""
    action: ActionType
    sandbox_id: str
    params: Dict[str, Any]
    
    def __init__(self, sender: str, receiver: str, action: ActionType, sandbox_id: str, params: Dict[str, Any]):
        data = {
            "action": action.value,
            "sandbox_id": sandbox_id,
            "params": params
        }
        super().__init__(MessageType.REQUEST, sender, receiver, data)
        self.action = action
        self.sandbox_id = sandbox_id
        self.params = params


@dataclass
class MCPResponse(MCPMessage):
    """MCP响应消息"""
    request_id: str
    success: bool
    result: Any
    error_message: Optional[str] = None
    
    def __init__(self, sender: str, receiver: str, request_id: str, success: bool, result: Any, error_message: Optional[str] = None):
        data = {
            "request_id": request_id,
            "success": success,
            "result": result,
            "error_message": error_message
        }
        super().__init__(MessageType.RESPONSE, sender, receiver, data)
        self.request_id = request_id
        self.success = success
        self.result = result
        self.error_message = error_message


class MCPProtocol:
    """MCP协议处理器"""
    
    def __init__(self, node_id: str):
        """初始化协议处理器
        
        Args:
            node_id: 当前节点ID
        """
        self.node_id = node_id
        self.message_history = []
    
    def create_request(self, receiver: str, action: ActionType, sandbox_id: str, params: Dict[str, Any]) -> MCPRequest:
        """创建请求消息"""
        return MCPRequest(
            sender=self.node_id,
            receiver=receiver,
            action=action,
            sandbox_id=sandbox_id,
            params=params
        )
    
    def create_response(self, receiver: str, request_id: str, success: bool, result: Any, error_message: Optional[str] = None) -> MCPResponse:
        """创建响应消息"""
        return MCPResponse(
            sender=self.node_id,
            receiver=receiver,
            request_id=request_id,
            success=success,
            result=result,
            error_message=error_message
        )
    
    def send_message(self, message: MCPMessage) -> None:
        """发送消息（记录到历史）"""
        self.message_history.append(message)
    
    def get_message_history(self) -> list[MCPMessage]:
        """获取消息历史"""
        return self.message_history.copy()
    
    def clear_history(self) -> None:
        """清空消息历史"""
        self.message_history.clear() 