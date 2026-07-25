using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Windows.Graphics;

namespace LightVideoEnhancer_WinUI;

public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.Resize(new SizeInt32(1280, 900));
        RootFrame.Navigate(typeof(MainPage));
    }

    public void ApplyTheme(ElementTheme theme)
    {
        // The Page is only part of the visual tree. Apply the theme to the
        // Window-owned root, title bar, Frame, and page so a dark Mica/fallback
        // layer cannot show through after switching to the light theme.
        WindowRoot.RequestedTheme = theme;
        AppTitleBar.RequestedTheme = theme;
        RootFrame.RequestedTheme = theme;
        if (RootFrame.Content is FrameworkElement content)
        {
            content.RequestedTheme = theme;
        }

        if (AppWindowTitleBar.IsCustomizationSupported())
        {
            AppWindow.TitleBar.PreferredTheme = theme switch
            {
                ElementTheme.Light => TitleBarTheme.Light,
                ElementTheme.Dark => TitleBarTheme.Dark,
                _ => TitleBarTheme.UseDefaultAppMode,
            };
        }
    }
}