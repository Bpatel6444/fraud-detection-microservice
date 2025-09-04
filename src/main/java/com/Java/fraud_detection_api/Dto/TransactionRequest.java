package com.Java.fraud_detection_api.Dto;

import java.time.Instant;
import java.util.Map;

public class TransactionRequest {
    private Double amount;
    private Instant timestamp;
    private String transactionId;
    private String merchantId;
    private String customerId;
    private Map<String, Double> pcaFeatures; // For V1-V28 features

    public TransactionRequest() {
    }

    public TransactionRequest(Double amount, Instant timestamp, String transactionId, 
                             String merchantId, String customerId, Map<String, Double> pcaFeatures) {
        this.amount = amount;
        this.timestamp = timestamp;
        this.transactionId = transactionId;
        this.merchantId = merchantId;
        this.customerId = customerId;
        this.pcaFeatures = pcaFeatures;
    }

    // Getters and setters
    public Double getAmount() {
        return amount;
    }

    public void setAmount(Double amount) {
        this.amount = amount;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Instant timestamp) {
        this.timestamp = timestamp;
    }

    public String getTransactionId() {
        return transactionId;
    }

    public void setTransactionId(String transactionId) {
        this.transactionId = transactionId;
    }

    public String getMerchantId() {
        return merchantId;
    }

    public void setMerchantId(String merchantId) {
        this.merchantId = merchantId;
    }

    public String getCustomerId() {
        return customerId;
    }

    public void setCustomerId(String customerId) {
        this.customerId = customerId;
    }

    public Map<String, Double> getPcaFeatures() {
        return pcaFeatures;
    }

    public void setPcaFeatures(Map<String, Double> pcaFeatures) {
        this.pcaFeatures = pcaFeatures;
    }
}