package com.tp4_omaressafi.test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {

        System.out.println("=== Phase 1 : Enregistrement des embeddings ===");

        // Création du parser PDF (Apache Tika)
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        // Chargement du fichier PDF
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, documentParser);

        // Découpage du document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Nombre de segments : " + segments.size());

        // Création du modèle d’embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Génération des embeddings pour tous les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("Nombre d'embeddings générés : " + embeddings.size());

        // Création du magasin d’embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Ajout des embeddings et segments associés
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Enregistrement des embeddings terminé avec succès !");

        System.out.println("\n=== Phase 2 : Recherche et réponse avec Gemini ===");

        // Clé API Gemini
        String GEMINI_API_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY manquante !");
        }

        // Création du modèle de chat Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .temperature(0.3)
                .modelName("gemini-2.5-flash")
                .build();

        // Création du ContentRetriever
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Ajout d'une mémoire de 10 messages
        var memory = MessageWindowChatMemory.withMaxMessages(10);

        // Création de l’assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .build();

        // Interaction console (multi-questions)
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Posez votre question (ou 'exit' pour quitter) :");
            while (true) {
                System.out.print("Vous : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("exit")) break;
                String reponse = assistant.chat(question);
                System.out.println("Gemini : " + reponse);
            }
        }
    }
}
