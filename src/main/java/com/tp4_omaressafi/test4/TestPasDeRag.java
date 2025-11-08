package com.tp4_omaressafi.test4;

import com.tp4_omaressafi.test1.Assistant;
import dev.langchain4j.data.document.*;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.query.Query;


import java.nio.file.*;
import java.util.*;
import java.util.Scanner;

public class TestPasDeRag {

    public static void main(String[] args) {

        System.out.println("=== Phase 1 : Ingestion du document RAG ===");

        // Parser + Chargement du PDF
        DocumentParser parser = new ApacheTikaDocumentParser();
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);

        //  Split + Embeddings
        var splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(doc);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        //  Stockage en mémoire
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        System.out.println(" Ingestion terminée avec " + segments.size() + " segments");

        System.out.println("\n=== Phase 2 : Chat avec routage conditionnel (RAG ou pas) ===");

        //  Modèle Gemini
        String GEMINI_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_KEY == null) throw new IllegalStateException(" GEMINI_KEY manquant !");
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        //  ContentRetriever
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        //  Création de la classe interne pour éviter le RAG
        class QueryRouterPourEviterRag implements QueryRouter {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                String question = "Est-ce que la requête '" + query.text()
                        + "' porte sur le 'RAG' (Retrieval Augmented Generation) ou le 'Fine Tuning' ? "
                        + "Réponds seulement par 'oui', 'non', ou 'peut-être'.";
                String reponse = model.chat(question).trim().toLowerCase();

                System.out.println(" Décision du QueryRouter : " + reponse);
                if (reponse.contains("non")) {
                    System.out.println(" Pas de RAG utilisé.");
                    return Collections.emptyList();
                } else {
                    System.out.println(" RAG activé.");
                    return List.of(retriever);
                }
            }
        }

        //  Instanciation du QueryRouter personnalisé
        QueryRouter queryRouter = new QueryRouterPourEviterRag();

        // Création du RetrievalAugmentor
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 9 Création de l’assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // Interaction console
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\n Vous : ");
                String q = scanner.nextLine();
                if (q.equalsIgnoreCase("exit")) break;
                String r = assistant.chat(q);
                System.out.println(" Gemini : " + r);
            }
        }
    }
}
