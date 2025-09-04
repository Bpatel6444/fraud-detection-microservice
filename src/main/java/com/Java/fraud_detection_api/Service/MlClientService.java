package com.Java.fraud_detection_api.Service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.HttpClientErrorException;

import java.util.HashMap;
import java.util.Map;

@Service
public class MlClientService {

    @Autowired
    private RestTemplate restTemplate;

    private static final String ML_API_URL = "http://localhost:5000/predict";

    public String checkTransactionWithML(Double amount, String timestamp, String transactionId, 
                                       String merchantId, String customerId, Map<String, Double> pcaFeatures) {
        try {
            // Build JSON request body with enhanced features
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("amount", amount);
            requestBody.put("timestamp", timestamp);
            requestBody.put("transactionId", transactionId);
            requestBody.put("merchantId", merchantId);
            requestBody.put("customerId", customerId);
            
            // Add PCA features if available
            if (pcaFeatures != null) {
                for (Map.Entry<String, Double> entry : pcaFeatures.entrySet()) {
                    requestBody.put(entry.getKey(), entry.getValue());
                }
            }

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

            // Send POST request to Python ML service
            ResponseEntity<String> response = restTemplate.postForEntity(ML_API_URL, request, String.class);

            return response.getBody();

        } catch (HttpClientErrorException e) {
            return "{\"error\": \"ML service error: " + e.getStatusCode() + " - " + e.getResponseBodyAsString() + "\"}";
        } catch (Exception e) {
            return "{\"error\": \"ML service unavailable: " + e.getMessage() + "\"}";
        }
    }
}