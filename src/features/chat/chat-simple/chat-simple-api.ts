import { userHashedId } from "@/features/auth/helpers";
import { CosmosDBChatMessageHistory } from "@/features/langchain/memory/cosmosdb/cosmosdb";
import { AI_NAME } from "@/features/theme/customise";
import { LangChainStream, StreamingTextResponse } from "ai";
import { ConversationChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { BufferWindowMemory } from "langchain/memory";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { initAndGuardChatSession } from "../chat-services/chat-thread-service";
import { PromptGPTProps } from "../chat-services/models";
import { transformConversationStyleToTemperature } from "../chat-services/utils";

export const ChatSimple = async (props: PromptGPTProps) => {
  const { lastHumanMessage, id, chatThread } = await initAndGuardChatSession(
    props
  );

  const { stream, handlers } = LangChainStream();

  const userId = await userHashedId();

  const chat = new ChatOpenAI({
    temperature: transformConversationStyleToTemperature(
      chatThread.conversationStyle
    ),
    streaming: true,
  });

  const memory = new BufferWindowMemory({
    k: 100,
    returnMessages: true,
    memoryKey: "history",
    chatHistory: new CosmosDBChatMessageHistory({
      sessionId: id,
      userId: userId,
    }),
  });

  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      // `-You are ${AI_NAME} who is a helpful AI Assistant.
      // - You will provide clear and concise queries, and you will respond with polite and professional answers.
      // - You will answer questions truthfully and accurately.`
      `- You are ${AI_NAME}, a helpful AI Assistant designed for internal use at RWE - an energy company.
      - Your primary function is to provide guidance on internal processes and procedures within RWE.
      - You will always speak in the context of the user being an employee and the chatbot being a part of the company. Use language that reflects this, such as referring to the company and its employees as "us."
      - Your responses should be aligned with the interests and policies of RWE, promoting a positive and cohesive work environment.
      - You will provide clear and concise queries, and you will respond with polite and professional answers.
      - You will answer questions truthfully and accurately, in accordance with RWE's guidelines and best practices.`
    ),
    new MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.fromTemplate("{input}"),
  ]);

  const chain = new ConversationChain({
    llm: chat,
    memory,
    prompt: chatPrompt,
  });

  chain.call({ input: lastHumanMessage.content }, [handlers]);

  return new StreamingTextResponse(stream);
};
