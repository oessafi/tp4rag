package com.tp4_omaressafi.test2;

import com.tp4_omaressafi.test1.Assistant;
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
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagNaifoptionnel {

    // Configuration du logger pour afficher les logs détaillés de LangChain4j
    private static void configureLogger() {
        System.out.println("Configuration du logger");
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Niveau détaillé
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger(); // Active le logging détaillé

        System.out.println("=== Phase 1 : Enregistrement des embeddings ===");

        // Création du parser PDF avec Apache Tika
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        // Chargement du fichier PDF à analyser
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, documentParser);

        // Découpage du document en segments pour l'analyse
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Nombre de segments extraits : " + segments.size());

        // Création du modèle d'embeddings pour transformer le texte en vecteurs
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Génération des embeddings pour tous les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("Nombre d'embeddings générés : " + embeddings.size());

        // Création d'un magasin d'embeddings en mémoire pour la recherche
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Ajout des embeddings et des segments correspondants dans le magasin
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Enregistrement des embeddings terminé avec succès !");

        System.out.println("\n=== Phase 2 : Recherche et réponse avec Gemini ===");

        // Récupération de la clé API Gemini depuis les variables d'environnement
        String GEMINI_API_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY manquante !");
        }

        // Création du modèle de chat Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .temperature(0.3) // Contrôle la créativité des réponses
                .logRequestsAndResponses(true) // Active le log des requêtes et réponses
                .modelName("gemini-2.5-flash") // Choix du modèle Gemini
                .build();

        // Création du retriever pour récupérer les segments les plus pertinents
        EmbeddingStoreContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3) // Nombre maximum de segments à retourner
                .minScore(0.5) // Score minimal pour filtrer les segments peu pertinents
                .build();

        // Création d'une mémoire pour stocker les 10 derniers messages
        var memory = MessageWindowChatMemory.withMaxMessages(10);

        // Création de l'assistant qui utilise le modèle Gemini et le retriever
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(retriever)
                .build();

        // Exemple de question pour tester le RAG
        String question = "Quelle est la signification de RAG ?";

        // Génération de l'embedding de la question
        Embedding embeddingQuestion = embeddingModel.embed(question).content();

        // Création de la requête de recherche dans le magasin d'embeddings
        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(embeddingQuestion)
                .maxResults(3)
                .minScore(0.5)
                .build();

        // Recherche des segments les plus pertinents pour la question
        EmbeddingSearchResult<TextSegment> embeddingSearchResult = embeddingStore.search(embeddingSearchRequest);

        // Affichage des segments récupérés avec leur score de similarité
        System.out.println("Segments pertinents avec leur score :");
        for (EmbeddingMatch<TextSegment> match : embeddingSearchResult.matches()) {
            System.out.println("Segment : " + match.embedded() + "\nScore : " + match.score() + "\n");
        }

        // Vérification : obtenir la réponse du modèle Gemini via RAG
        String reponse = assistant.chat(question);
        System.out.println("Réponse du modèle Gemini (avec RAG) :\n" + reponse);
    }
}
