void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    setupGimbal();
    buildConnectors();
    fragColor = vec4(bgLight(fragCoord / iResolution.xy), 1.0);
}
