package com.app.oneklikai.exceptions;

import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.ConstraintViolationException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ProblemDetail;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.HandlerMethodValidationException;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

import java.util.stream.Collectors;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    private ProblemDetail buildProblem(HttpStatus status, String detail, String title, HttpServletResponse response) {
        response.setContentType(MediaType.APPLICATION_PROBLEM_JSON_VALUE);
        ProblemDetail problemDetail = ProblemDetail.forStatusAndDetail(status, detail);
        problemDetail.setTitle(title);
        return problemDetail;
    }

    @ExceptionHandler(ResourceAlreadyExistsException.class)
    public ProblemDetail handleConflict(ResourceAlreadyExistsException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.CONFLICT, ex.getMessage(), "Conflict", response);
    }

    @ExceptionHandler(UnauthorizedException.class)
    public ProblemDetail handleUnauthorized(UnauthorizedException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.UNAUTHORIZED, ex.getMessage(), "Unauthorized", response);
    }

    @ExceptionHandler(Exception.class)
    public ProblemDetail handleGeneralError(Exception ex, HttpServletResponse response) {
        log.error("An unexpected error occurred", ex);
        return buildProblem(HttpStatus.INTERNAL_SERVER_ERROR, "An unexpected error occurred", "Internal Server Error", response);
    }

    @ExceptionHandler(BadRequestException.class)
    public ProblemDetail handleBadRequest(BadRequestException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.BAD_REQUEST, ex.getMessage(), "Bad Request", response);
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ProblemDetail handleBodyValidation(MethodArgumentNotValidException ex, HttpServletResponse response) {
        String detail = ex.getBindingResult().getFieldErrors().stream()
                .map(fe -> fe.getField() + ": " + fe.getDefaultMessage())
                .collect(Collectors.joining("; "));
        return buildProblem(HttpStatus.BAD_REQUEST, detail.isBlank() ? "Validation failed" : detail, "Validation Failed", response);
    }

    @ExceptionHandler(HandlerMethodValidationException.class)
    public ProblemDetail handleParamValidation(HandlerMethodValidationException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.BAD_REQUEST,
                ex.getReason() == null ? "Invalid request parameters" : ex.getReason(), "Validation Failed", response);
    }

    @ExceptionHandler(ConstraintViolationException.class)
    public ProblemDetail handleConstraintViolation(ConstraintViolationException ex, HttpServletResponse response) {
        String detail = ex.getConstraintViolations().stream()
                .map(v -> v.getPropertyPath() + ": " + v.getMessage())
                .collect(Collectors.joining("; "));
        return buildProblem(HttpStatus.BAD_REQUEST, detail.isBlank() ? "Validation failed" : detail, "Validation Failed", response);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ProblemDetail handleUnreadableBody(HttpServletResponse response) {
        return buildProblem(HttpStatus.BAD_REQUEST, "Malformed or unreadable request body", "Bad Request", response);
    }

    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ProblemDetail handleTypeMismatch(MethodArgumentTypeMismatchException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.BAD_REQUEST,
                "Invalid value for parameter '" + ex.getName() + "'", "Bad Request", response);
    }

    @ExceptionHandler(NotFoundException.class)
    public ProblemDetail handleNotFound(NotFoundException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.NOT_FOUND, ex.getMessage(), "Not Found", response);
    }

    @ExceptionHandler(ForbiddenException.class)
    public ProblemDetail handleForbidden(ForbiddenException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.FORBIDDEN, ex.getMessage(), "Forbidden", response);
    }

    @ExceptionHandler(NoResourceFoundException.class)
    public ProblemDetail handleNoResourceFound(HttpServletResponse response) {
        return buildProblem(HttpStatus.NOT_FOUND, "The requested resource was not found.", "Not Found", response);
    }

    @ExceptionHandler(IllegalStateException.class)
    public ProblemDetail handleIllegalState(IllegalStateException ex, HttpServletResponse response) {
        return buildProblem(HttpStatus.BAD_REQUEST, ex.getMessage(), "Invalid Application State", response);
    }

}
